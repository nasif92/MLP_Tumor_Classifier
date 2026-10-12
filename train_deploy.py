"""
Trains ONE pooled model on all subtypes (ER, PR, Ki67 by default) and saves it
for deployment. Validation runs on ALL reference sets (ER + Ki67 by default),
reported pooled AND per subtype.

Usage:
    # Pooled model on ER+PR+Ki67, validated on every reference set (annotation labels):
    python train_deploy.py --out pooled_model.pt --cache_dir ./cache
    python train_deploy.py --feat_dir . --ann_dir . --holdout_frac 0.1 --val_source holdout  # sanity check
    # Train on phase 1 only, evaluate on phase 2 as a fully separate test set:
    python train_deploy.py --feat_dir <phase1_feat_dir> --ann_dir <phase1_ann_dir> \\
        --phase2_feat_dir <phase2_feat_dir> --phase2_ann_dir <phase2_ann_dir> \\
        --holdout_frac 0 --out phase1_plus_phase2.pt

    # Single-slide training (e.g. one small dataset):
    python3 train_deploy.py --feat_dir <dir_with_one_slide> --ann_dir <ann_dir> --out one_slide.pt

    # Feature-group Mixture-of-Experts (see moe.py), stain-conditioned gate:
    python train_deploy.py --arch moe --feature_set default_derived --out moe.pt --cache_dir ./cache
    # Matched baseline for the same features:
    python train_deploy.py --arch mlp --feature_set default_derived --out mlp.pt --cache_dir ./cache
"""
import argparse
import gzip
import hashlib
import json
import os
import re
import sys
from collections import Counter
import config as cf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import moe

# Curated reference/validation slides: one detections dir + one annotations
# dir PER SUBTYPE, paths from config.py. A plain run validates the pooled
# model on every subtype listed here whose folders exist. Each subtype is
# loaded only from its own pair of dirs (so a shared/combined folder can't
# pull in - or double-count - another subtype's slides).
REFERENCE_SETS = {
    "ER":   {"detections": cf.ref_set_er_all_detections,
             "annotations": cf.ref_set_ann_er},
    "PR":   {"detections": cf.ref_set_no_cells,
             "annotations": cf.ref_set_ann_pr},
    "Ki67": {"detections": cf.ref_set_ki67_all_detections,
             "annotations": cf.ref_set_ann_ki67},
}
REFERENCE_TEST_DIRS = [s["detections"] for s in REFERENCE_SETS.values() if s["detections"]]
REFERENCE_ANN_DIRS = [s["annotations"] for s in REFERENCE_SETS.values() if s["annotations"]]
# Kept for older scripts that import the single-dir names.
REFERENCE_TEST_DIR = cf.ref_set_ki67_all_detections
REFERENCE_ANN_DIR = cf.ref_set_ann_ki67

NON_FEATURE_COLS = {
    "wsi_name", "nucleus_id", "det_index", "det_uuid", "cx_wsi", "cy_wsi",
    "label", "is_ground_truth",
    "containing_annotation_ids", "containing_annotation_classes",
    "annotation_nesting_depth",
}

# Intensity measurements (stain-dependent - the ones a staining/scanner batch
# effect actually shows up in) vs. everything else (morphology - stable
# across slides). Matches at any resolution prefix: "X", "ROI: 0.50 um per
# pixel: X", "Smoothed: 50 um: X", "Smoothed: 50 um: ROI: 0.50 um per pixel: X".
INTENSITY_FEATURE_RE = re.compile(r"(Hematoxylin|DAB|OD Sum):", re.IGNORECASE)

# Feature sets live in config.FEATURE_SETS (one place, alongside the other
# paths/constants) - see config.py for "default" (excludes "Smoothed: ..."
# measurements) and "all". --feature_set picks one by name; --feature_list
# still allows a one-off external file when you don't want to edit config.py.
def load_feature_list_file(path):
    """Read a plain one-name-per-line .txt, or a JSON list/dict with an
    "active" key, for a one-off feature set outside config.FEATURE_SETS."""
    import json
    if path.endswith(".json"):
        data = json.load(open(path))
        names = data["active"] if isinstance(data, dict) else data
    else:
        names = [ln.strip() for ln in open(path) if ln.strip() and not ln.startswith("#")]
    return list(dict.fromkeys(names))  # de-dup, keep order


# Nucleus loading lives in nuclei_io.py, shared with extract_crops.py so both
# select exactly the same nuclei/labels/centroids. The old private names are
# kept as aliases for any script that still imports them from here.
from nuclei_io import (
    LOADER_VERSION,
    infer_subtype_from_name as _infer_subtype_from_name,
    load_geojson_detections as _load_geojson_any,
    polygon_centroid as _polygon_centroid,
    find_annotation_file as _find_annotation_file,
    labels_from_classification as _labels_from_classification,
    iter_slides,
    load_slide_nuclei,
)


def _load_slide_features_and_labels(feat_path, ann_path, slide_name):
    df, _ = load_slide_nuclei(feat_path, ann_path, slide_name)
    return df


def _load_geojson_root(feat_dir, ann_dir, subtypes, feat_pattern=".geojson.gz",
                        label="data", use_classification=False):
    if not feat_dir:
        return None
    frames = []
    for rec in iter_slides(feat_dir, ann_dir, subtypes, feat_pattern, use_classification):
        slide_name, st = rec["slide_name"], rec["subtype"]
        if rec["skip"]:
            print(f"  {slide_name}: {rec['skip']} - skipping")
            continue
        try:
            df_slide = _load_slide_features_and_labels(rec["feat_path"], rec["ann_path"], slide_name)
        except Exception as e:
            print(f"  {slide_name}: FAILED to load {os.path.basename(rec['feat_path'])} "
                  f"({type(e).__name__}: {e}) - skipping")
            continue
        if df_slide is None or df_slide.empty:
            continue
        df_slide = pd.concat([df_slide, pd.DataFrame({
            "subtype": st,
            "slide_uid": st + ":" + df_slide["wsi_name"].astype(str),
            "ann_source_dir": rec["ann_source_dir"],
        }, index=df_slide.index)], axis=1)
        frames.append(df_slide)
        print(f"  {st} ({label}): {slide_name} - {len(df_slide)} nuclei")
    return pd.concat(frames, ignore_index=True) if frames else None


def _newest_mtime(root, suffix=None):
    newest = 0.0
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if suffix is not None and not fn.endswith(suffix):
                continue
            try:
                newest = max(newest, os.path.getmtime(os.path.join(dirpath, fn)))
            except OSError:
                pass
    return newest


def _ann_dir_key(ann_dir):
    """Normalize a single dir or a list of dirs into one stable string, for
    cache-key hashing and mtime checks."""
    if not ann_dir:
        return ""
    dirs = [ann_dir] if isinstance(ann_dir, str) else list(ann_dir)
    return "|".join(sorted(dirs))


def _cache_key(feat_dir, ann_dir, subtypes, feat_pattern, use_classification):
    raw = "|".join([feat_dir or "", _ann_dir_key(ann_dir), ",".join(sorted(subtypes)),
                     feat_pattern, str(use_classification),
                     f"loader{LOADER_VERSION}"])
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _load_geojson_root_cached(feat_dir, ann_dir, subtypes, cache_dir=None,
                               feat_pattern=".geojson.gz", label="data",
                               use_classification=False):
    """Same as _load_geojson_root, cached to disk by mtime-based invalidation."""
    if not feat_dir:
        return None
    if not cache_dir:
        return _load_geojson_root(feat_dir, ann_dir, subtypes, feat_pattern,
                                   label, use_classification)

    os.makedirs(cache_dir, exist_ok=True)
    key = _cache_key(feat_dir, ann_dir, subtypes, feat_pattern, use_classification)
    cache_path = os.path.join(cache_dir, f"{key}.pkl")

    newest_source_mtime = _newest_mtime(feat_dir, suffix=feat_pattern)
    if ann_dir:
        ann_dirs = [ann_dir] if isinstance(ann_dir, str) else list(ann_dir)
        for d in ann_dirs:
            newest_source_mtime = max(
                newest_source_mtime,
                max(_newest_mtime(d, suffix=".geojson"),
                    _newest_mtime(d, suffix=".geojson.gz")))

    if os.path.exists(cache_path) and os.path.getmtime(cache_path) >= newest_source_mtime:
        print(f"  ({label}) loading from cache: {cache_path}")
        return pd.read_pickle(cache_path)

    df = _load_geojson_root(feat_dir, ann_dir, subtypes, feat_pattern, label, use_classification)
    if df is not None:
        df.to_pickle(cache_path)
        print(f"  ({label}) cached -> {cache_path}")
    return df


def load_test_sets(test_feat_dirs, test_ann_dirs, subtypes, cache_dir=None,
                   feat_pattern=".geojson.gz", label="test"):
    """Load and concatenate EVERY reference/test detections dir.
    test_ann_dirs: list of annotation dirs searched for every slide (labels =
    annotation join, i.e. pathologist ground truth); empty -> each
    detection's own in-file classification is used as its label instead."""
    test_ann_dirs = [d for d in (test_ann_dirs or []) if d]
    use_cls = not test_ann_dirs
    frames = []
    for d in test_feat_dirs or []:
        if not d:
            continue
        name = os.path.basename(os.path.normpath(os.path.dirname(os.path.normpath(d)))) \
            or os.path.basename(os.path.normpath(d))
        df_d = _load_geojson_root_cached(d, "" if use_cls else test_ann_dirs, subtypes,
                                          cache_dir=cache_dir, feat_pattern=feat_pattern,
                                          label=f"{label}: {name}",
                                          use_classification=use_cls)
        if df_d is None or df_d.empty:
            print(f"  ({label}) nothing loaded from {d}")
            continue
        frames.append(df_d)
    return pd.concat(frames, ignore_index=True) if frames else None


def load_reference_sets(subtypes, use_annotations=True, cache_dir=None,
                        feat_pattern=".geojson.gz", label="reference"):
    """Load the reference set of each requested subtype from REFERENCE_SETS,
    each subtype from ITS OWN detections + annotations dirs only."""
    frames = []
    for st in subtypes:
        ref = REFERENCE_SETS.get(st)
        det = ref and ref.get("detections")
        ann = ref and ref.get("annotations")
        if not det:
            print(f"  ({label}) {st}: no reference set configured in config.py - skipped")
            continue
        if not os.path.isdir(det):
            print(f"  ({label}) {st}: detections dir not found - skipped: {det}")
            continue
        if use_annotations and (not ann or not os.path.isdir(ann)):
            print(f"  ({label}) {st}: annotations dir not found - skipped: {ann}")
            continue
        df_st = _load_geojson_root_cached(det, ann if use_annotations else "", [st],
                                           cache_dir=cache_dir, feat_pattern=feat_pattern,
                                           label=f"{label}: {st}",
                                           use_classification=not use_annotations)
        if df_st is None or df_st.empty:
            print(f"  ({label}) {st}: no labeled nuclei loaded from {det}")
            continue
        frames.append(df_st)
    return pd.concat(frames, ignore_index=True) if frames else None


def load_eval_set(args, subtypes, label):
    """Validation/test set for train_deploy: the per-subtype REFERENCE_SETS by
    default, or the explicit --test_feat_dir/--test_ann_dir dirs if given.
    --test_ann_dir given with no value -> in-file classification labels."""
    ann = args.test_ann_dir
    in_file = ann is not None and not [d for d in ann if d]
    if not args.test_feat_dir:
        print(f"\n{label}: reference sets for {list(subtypes)}, labels = "
              + ("in-file classification" if in_file else "annotation join"))
        return load_reference_sets(subtypes, use_annotations=not in_file,
                                   cache_dir=args.cache_dir,
                                   feat_pattern=args.test_feat_pattern, label=label)
    ann_dirs = [] if in_file else ([d for d in ann if d] if ann else REFERENCE_ANN_DIRS)
    print(f"\n{label}: {len(args.test_feat_dir)} dir(s), labels = "
          + ("in-file classification" if not ann_dirs else "annotation join"))
    return load_test_sets(args.test_feat_dir, ann_dirs, subtypes, cache_dir=args.cache_dir,
                          feat_pattern=args.test_feat_pattern, label=label)


# QuPath class names to write back, keyed by the COLLAPSED training label.
QUPATH_NAMES = {
    "Tumor": "Tumor",
    "Non Tumor": "Non Tumor",
    "Normal": "Normal-Glands",
    "Ghost": "Ghost",
}


class MLP(nn.Module):
    def __init__(self, n_features, n_classes, hidden=(48, 24, 12, 6), dropout=0.3):
        super().__init__()
        layers, prev = [], n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)
        self.feat_dim = hidden[-1]

    def forward(self, x, subtype=None):  # subtype accepted (and ignored) so MLP/MoE share call sites
        return self.net(x)

    def features(self, x, subtype=None):
        """Shared-trunk representation (everything before the final Linear).
        Used by --domain_adversarial to attach a domain head; unused, and
        adding nothing to model_state, when that flag is off."""
        return self.net[:-1](x)

    def classify(self, feats):
        """Final Linear applied to a features() output - forward(x) ==
        classify(features(x)). (Named to match MoEClassifier.classify.)"""
        return self.net[-1](feats)

    head = classify  # backwards-compatible alias


class _GradReverseFn(torch.autograd.Function):
    """Identity on the forward pass; negates (and scales) the gradient on
    the backward pass. This is the whole mechanism behind
    --domain_adversarial: the shared trunk is pushed by the task loss to
    classify Tumor/Non-Tumor well, and pushed by the REVERSED domain-head
    gradient to make slide/site identity HARDER to predict from its
    features - so what survives is task-relevant, not slide-identifying."""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return _GradReverseFn.apply(x, lambd)


def domain_lambda_schedule(progress, lambda_max):
    """Standard DANN ramp: 0 at the start of training, smoothly up to
    lambda_max, so the adversarial pressure doesn't destabilize the task
    loss before the trunk has learned anything useful yet. progress in [0,1]."""
    return lambda_max * (2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0)


_SITE_RE = re.compile(r"^[A-Za-z]+")


def infer_site_from_slide(wsi_name):
    """Site/institution code from a slide name, e.g. "MIS24-007597_KI67_0"
    -> "MIS". Used as the (coarser, better-supported) domain-head target
    when --domain_level site."""
    m = _SITE_RE.match(str(wsi_name))
    return m.group() if m else "unknown"


class Metrics:
    """Tracks one split's (train/val) loss, accuracy, and per-class accuracy."""

    def __init__(self, name):
        self.name = name
        self.epoch = -1
        self.loss = None
        self.acc = None
        self.per_class_acc = {}
        self.max_acc = -1.0
        self.min_loss = float("inf")
        self.max_acc_epoch = 0
        self.min_loss_epoch = 0

    def update(self, epoch, loss, acc, per_class_acc=None):
        self.epoch = epoch
        self.loss = loss
        self.acc = acc
        self.per_class_acc = per_class_acc or {}
        improved = acc > self.max_acc
        if improved:
            self.max_acc = acc
            self.max_acc_epoch = epoch
        if loss < self.min_loss:
            self.min_loss = loss
            self.min_loss_epoch = epoch
        return improved

    def to_writer(self, writer):
        writer.add_scalar(f"{self.name}/loss", self.loss, self.epoch)
        writer.add_scalar(f"{self.name}/acc", self.acc, self.epoch)
        for cls_name, cls_acc in self.per_class_acc.items():
            writer.add_scalar(f"{self.name}/acc_{cls_name}", cls_acc, self.epoch)

    def to_str(self):
        return (f"{self.name}_loss={self.loss:.4f}  {self.name}_acc={self.acc:.4f}  "
                f"(best acc {self.max_acc:.4f} @ epoch {self.max_acc_epoch}, "
                f"best loss {self.min_loss:.4f} @ epoch {self.min_loss_epoch})")


def per_class_accuracy(y_true, y_pred, classes):
    """Recall per class (diagonal of the confusion matrix / row support)."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    support = cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(support > 0, np.diag(cm) / np.maximum(support, 1), 0.0)
    return {c: float(r) for c, r in zip(classes, recall)}, cm


def load_and_label(df_ext, classes, feat_cols=None, medians=None, mu=None, sd=None):
    """Apply collapse_label + class filtering, and optionally the training
    run's fixed feature columns/medians/mu/sd (for a val/test set)."""
    df_ext["_label"] = df_ext["label"].apply(cf.collapse_label)
    n_dropped = int(df_ext["_label"].isna().sum())
    if n_dropped:
        print(f"  dropped {n_dropped} rows with unmapped/excluded raw labels:")
        for k, v in df_ext.loc[df_ext["_label"].isna(), "label"].value_counts(dropna=False).items():
            print(f"    '{k}': {v}")
    df_ext = df_ext.dropna(subset=["_label"])
    df_ext = df_ext[df_ext["_label"].isin(classes)].reset_index(drop=True)
    if df_ext.empty or feat_cols is None:
        return df_ext, None, None
    moe.add_derived_features(df_ext)
    missing = [c for c in feat_cols if c not in df_ext.columns]
    if missing:
        print(f"  WARNING: {len(missing)} training feature(s) absent from this data - "
              f"median-filled on every nucleus: {missing[:8]}")
    X = df_ext.reindex(columns=feat_cols).astype(np.float64)
    X = X.fillna(medians[feat_cols])
    # Same +-20 sd clip as the training inputs.
    X_s = np.clip((X.to_numpy() - mu) / sd, -20, 20).astype(np.float32)
    c2i = {c: i for i, c in enumerate(classes)}
    y = df_ext["_label"].map(c2i).to_numpy().copy()
    return df_ext, X_s, y


def report_eval(name, y_true, y_pred, classes):
    from sklearn.metrics import classification_report, f1_score
    print(f"\n=== {name} ===")
    print(f"acc={np.mean(y_pred == y_true):.4f}  "
          f"macro_f1={f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(classification_report(y_true, y_pred, labels=list(range(len(classes))),
                                target_names=classes, zero_division=0))
    _, cm = per_class_accuracy(y_true, y_pred, classes)
    print("Confusion (rows=true, cols=pred): " + "  ".join(classes))
    for i, row in enumerate(cm):
        print(f"{classes[i]:>8}: " + " ".join(f"{v:7d}" for v in row))


def report_eval_by_subtype(name, y_true, y_pred, classes, subtypes_arr, slides_arr):
    """One report per subtype, plus a POOLED report if there is more than
    one subtype (with only one subtype, POOLED would just repeat it)."""
    subtypes_present = sorted(set(subtypes_arr))
    if len(subtypes_present) > 1:
        report_eval(f"{name} - POOLED", y_true, y_pred, classes)
    for st in subtypes_present:
        m = subtypes_arr == st
        report_eval(f"{name} - {st} ({int(m.sum())} nuclei, "
                    f"{len(set(slides_arr[m]))} slides)", y_true[m], y_pred[m], classes)
    ok = pd.Series(y_pred == y_true)
    per = pd.DataFrame({
        "subtype": pd.Series(subtypes_arr).groupby(slides_arr).first(),
        "n": ok.groupby(slides_arr).size(),
        "acc": ok.groupby(slides_arr).mean(),
    })
    print(f"\n{name} - per slide:")
    print(per.sort_values(["subtype", "acc"]).round(3).to_string())
    print("slide-mean acc by subtype: " + ", ".join(
        f"{st}={g['acc'].mean():.4f}" for st, g in per.groupby("subtype")))


def slide_overlap_check(train_slides, test_slides, what="validation"):
    overlap = sorted(set(train_slides) & set(test_slides))
    if overlap:
        print(f"  WARNING: {len(overlap)} {what} slide(s) are ALSO in the training set - "
              f"their scores are not independent: {overlap[:10]}")
    else:
        print(f"  no slide overlap between training and {what} sets")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", default=cf.all_features,
                    help="Root directory to walk for <slide>-feat-no_cells.geojson(.gz) exports.")
    ap.add_argument("--feat_pattern", default="-feat-no_cells.geojson.gz",
                    help="Filename suffix identifying a slide's feature file under "
                         "--feat_dir/--phase2_feat_dir (training data). NOTE: this is "
                         "different from --test_feat_pattern - the reference/test set "
                         "uses plain '<slide>.geojson.gz' naming with no '-feat-no_cells' "
                         "infix, since it's already pre-classified in-file rather than "
                         "exported from the feature-extraction pipeline.")
    ap.add_argument("--ann_dir", nargs="+", default=[cf.phase1, cf.phase2],
                    help="One or more directories containing <slide>.geojson(.gz) "
                         "annotation files - searched in order, first match per "
                         "slide wins. Defaults to phase1_annotations + "
                         "phase2_annotations combined (no special phase weighting "
                         "unless --phase2_slides is also given).")
    ap.add_argument("--phase2_feat_dir", default=None,
                    help="Feature directory for phase 2 data, combined with --feat_dir at "
                         "training time. A slide only counts as phase 2 if it has a matching "
                         "annotations file under --phase2_ann_dir. OFF by default - phase1/"
                         "phase2 splitting isn't needed right now; pass this explicitly to "
                         "re-enable combining a second data source.")
    ap.add_argument("--phase2_ann_dir", default=None,
                    help="Annotation directory matching --phase2_feat_dir.")
    ap.add_argument("--subtypes", nargs="+", default=cf.DEFAULT_SUBTYPES,
                    help="Subtypes pooled into ONE training set (default: all of "
                         "config.DEFAULT_SUBTYPES, i.e. ER PR Ki67).")
    ap.add_argument("--subtype_balance", type=float, default=0.0,
                    help="Extra sampling weight 1/(n_subtype_nuclei ** p) so no single "
                         "subtype dominates the pooled model. 0 = off (sample by class "
                         "only, as before); 0.5 = partial; 1.0 = every subtype seen "
                         "equally often.")
    ap.add_argument("--domain_adversarial", action="store_true",
                    help="Add a gradient-reversal domain head on the shared trunk, "
                         "predicting slide/site identity (--domain_level) from TRAINING "
                         "nuclei only. The reversed gradient pushes the trunk to make that "
                         "harder to predict, so it keeps task-relevant signal while "
                         "discarding whatever distinguishes slides/sites. Needs no "
                         "unlabeled data - it reuses the existing training slide IDs. The "
                         "domain head itself is not saved (unused at inference); the "
                         "checkpoint's model_state is unaffected either way.")
    ap.add_argument("--domain_level", choices=["slide", "site"], default="site",
                    help="What the domain head predicts. 'site' (default) groups slides by "
                         "the institution code in their name (e.g. 'MIS24-...' -> 'MIS') - "
                         "fewer, better-supported classes, and matches the site-level shift "
                         "seen in diag_ki67. 'slide' is finer-grained but each class has "
                         "only that one slide's nuclei.")
    ap.add_argument("--domain_lambda_max", type=float, default=1.0,
                    help="Max gradient-reversal strength, reached via a smooth ramp over "
                         "training (0 at epoch 1). Higher = trunk works harder to hide "
                         "slide/site identity, at some risk of discarding real signal that "
                         "happens to correlate with site.")
    ap.add_argument("--domain_hidden", type=int, default=32,
                    help="Hidden width of the (small, 2-layer) domain head.")
    ap.add_argument("--feature_set", default="all", choices=sorted(cf.FEATURE_SETS),
                help=f"Named feature set from config.FEATURE_SETS. Available: {sorted(cf.FEATURE_SETS)}.")
    ap.add_argument("--feature_list", default=None,
                    help="Path to a one-off feature list file (plain text, one name per "
                         "line, or JSON) instead of a config.FEATURE_SETS entry.")
    ap.add_argument("--classes", nargs="+", default=["Tumor", "Non Tumor"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout_frac", type=float, default=0.0,
                    help="Hold out this fraction of SLIDES for a sanity check. "
                         "Use 0 for the final deployment model. Ignored when "
                         "--val_source test, or when only one slide is available.")
    ap.add_argument("--pool_reference_into_train", action="store_true",
                    help="Merge the annotated reference/test slides (config.REFERENCE_SETS) "
                         "INTO the training pool before any split, so --val_source holdout "
                         "draws its random held-out slides from training AND reference "
                         "slides together - i.e. a random mix-everything split, with no "
                         "site held out on purpose. Use this to check whether the "
                         "train-vs-reference gap seen with --val_source test is a real "
                         "generalization (site-shift) problem, or just normal train/test "
                         "noise: if random-holdout accuracy here is much higher than the "
                         "--val_source test accuracy on the same slides, that is evidence "
                         "for a genuine site-generalization gap, not just a hard model fit. "
                         "Only meaningful together with --val_source holdout.")
    ap.add_argument("--val_source", choices=["holdout", "test"], default="test",
                    help="Where periodic validation comes from. 'holdout' carves out "
                         "--holdout_frac of the training slides. 'test' (default) uses "
                         "--test_feat_dir/--test_ann_dir instead, training on ALL slides.")
    ap.add_argument("--val_select", choices=["macro", "pooled"], default="macro",
                    help="Validation loss used for best-checkpoint selection, early "
                         "stopping and the plateau LR schedule. 'macro' (default) = mean "
                         "of the per-subtype losses, so a subtype with many validation "
                         "nuclei can't drown out the others; 'pooled' = loss over all "
                         "validation nuclei together.")
    # ---------------------------------------------------------- architecture
    ap.add_argument("--arch", choices=["mlp", "moe"], default="mlp",
                    help="'mlp' = the original single MLP (--hidden/--dropout). 'moe' = "
                         "feature-group mixture of experts from moe.py: one small expert per "
                         "feature family (shape/size/intensity/texture[/context/embedding]), "
                         "a stain-conditioned softmax gate, and a shared head.")
    ap.add_argument("--moe_dim", type=int, default=32, help="Per-expert embedding size d.")
    ap.add_argument("--moe_expert_hidden", type=int, default=32,
                    help="Expert hidden width (auto-widened to 2x inputs for big groups, e.g. texture).")
    ap.add_argument("--moe_gate_hidden", type=int, default=64)
    ap.add_argument("--moe_head_hidden", type=int, default=32)
    ap.add_argument("--moe_dropout", type=float, default=0.2,
                    help="Dropout inside experts/gate/head (MoE only; --dropout is the MLP's).")
    ap.add_argument("--moe_expert_dropout", type=float, default=0.15,
                    help="Per-nucleus probability of masking a whole expert during training.")
    ap.add_argument("--moe_aux_weight", type=float, default=0.2,
                    help="Weight of the per-expert auxiliary CE losses (0 = off).")
    ap.add_argument("--moe_balance_weight", type=float, default=0.01,
                    help="Weight of the gate load-balancing term KL(usage||uniform) (0 = off).")
    ap.add_argument("--moe_bag", action="store_true",
                    help="Online bagging on the aux losses: each expert sees its own "
                         "Poisson(1) bootstrap weights, for extra diversity (Windeatt 2008).")
    ap.add_argument("--moe_gate_temp", type=float, default=1.0,
                    help="Gate softmax temperature (>1 = softer mixing, <1 = sharper routing).")
    ap.add_argument("--moe_no_subtype_gate", action="store_true",
                    help="Don't feed the stain (ER/PR/Ki67) to the gate.")
    ap.add_argument("--moe_subtype_drop", type=float, default=0.2,
                    help="Probability of hiding the stain from the gate during training, so "
                         "inference without a stain still works.")
    ap.add_argument("--moe_drop_groups", nargs="*", default=[],
                    help="Expert groups to leave out entirely (ablations), e.g. --moe_drop_groups texture.")
    ap.add_argument("--hidden", nargs="+", type=int, default=[96,48, 24, 12, 6])
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=10000)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--reweight_power", type=float, default=0.5)
    ap.add_argument("--phase2_slides", nargs="+", default=None,
                    help="Slide names (wsi_name) to up-weight as phase 2 during retraining, "
                         "even if loaded from --feat_dir.")
    ap.add_argument("--phase2_weight", type=float, default=1.0,
                    help="Sampling-weight multiplier for phase-2 slides. Only has any "
                         "effect when --phase2_feat_dir or --phase2_slides is actually "
                         "given - kept at a ready-to-use default (1.5x) for whenever "
                         "phase-2 data is reintroduced.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default='',
                    help="Force a specific device, e.g. 'cuda', 'cpu'. Auto-detects otherwise.")
    ap.add_argument("--tb_dir", default=None,
                    help="TensorBoard log directory. Defaults to '<out>_tb/'.")
    ap.add_argument("--no_tb", action="store_true", help="Disable TensorBoard logging.")
    ap.add_argument("--ckpt_best", default=None,
                    help="Where to save the best-epoch checkpoint. Defaults to '<out>.best'.")
    ap.add_argument("--val_every", type=int, default=1,
                    help="Evaluate on the validation set every N epochs.")
    ap.add_argument("--early_stop", action="store_true",
                    help="Stop once val loss hasn't improved for --patience epochs, and "
                         "reload the best epoch's weights before saving --out. Requires a "
                         "validation set (--holdout_frac > 0 or --val_source test).")
    ap.add_argument("--patience", type=int, default=30,
                    help="Epochs without val-loss improvement before stopping.")
    ap.add_argument("--lr_schedule", choices=["none", "plateau", "cosine"], default="plateau",
                    help="plateau halves --lr on val-loss stall (needs a validation set); "
                         "cosine decays --lr smoothly over --epochs regardless of holdout.")
    ap.add_argument("--test_feat_dir", nargs="+", default=None,
                    help="Detections dir(s) to evaluate on instead of the default "
                         "REFERENCE_SETS (ER, PR, Ki67 reference sets from config.py, each "
                         "subtype loaded from its own dirs). NEVER used for training. "
                         "Evaluation reuses this run's feature columns/medians/mu/sd.")
    ap.add_argument("--test_ann_dir", nargs="*", default=None,
                    help="Annotation dir(s) for --test_feat_dir (default: the reference "
                         "annotation dirs). Labels = annotation region each nucleus falls "
                         "in (pathologist ground truth). Pass '--test_ann_dir' with no "
                         "value to use each detection's own in-file classification.")
    ap.add_argument("--test_feat_pattern", default=".geojson.gz",
                    help="Filename suffix identifying a slide's feature file under "
                         "--test_feat_dir.")
    ap.add_argument("--test_subtypes", nargs="+", default=None,
                    help="Subtypes to load from --test_feat_dir. Defaults to --subtypes.")
    ap.add_argument("--cache_dir", default=None,
                    help="Directory to cache loaded+labeled GeoJSON dataframes as pickles, "
                         "auto-invalidated by source file mtimes. Off by default.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = _load_geojson_root_cached(args.feat_dir, args.ann_dir, args.subtypes,
                                    cache_dir=args.cache_dir, feat_pattern=args.feat_pattern,
                                    label="Reference")
    if df is None:
        raise SystemExit("No data found under --feat_dir/--ann_dir.")

    df["_phase_source"] = 1
    if args.phase2_feat_dir:
        df2 = _load_geojson_root_cached(args.phase2_feat_dir, args.phase2_ann_dir, args.subtypes,
                                         cache_dir=args.cache_dir, feat_pattern=args.feat_pattern,
                                         label="phase 2")
        if df2 is not None:
            df2["_phase_source"] = 2
            df = pd.concat([df, df2], ignore_index=True)

    if args.pool_reference_into_train:
        ref_df = load_reference_sets(args.subtypes, use_annotations=True,
                                     cache_dir=args.cache_dir,
                                     feat_pattern=args.test_feat_pattern,
                                     label="reference (pooled into train)")
        if ref_df is None or ref_df.empty:
            print("  --pool_reference_into_train: no reference data found - ignored")
        else:
            ref_df["_phase_source"] = 1  # not phase-2 by default; same weight as training
            overlap = set(df["wsi_name"]) & set(ref_df["wsi_name"])
            if overlap:
                print(f"  NOTE: {len(overlap)} slide(s) already in --feat_dir are ALSO in "
                      f"the reference set - they will appear twice in the pool: {sorted(overlap)[:5]}")
            df = pd.concat([df, ref_df], ignore_index=True)
            print(f"  --pool_reference_into_train: added {ref_df['wsi_name'].nunique()} "
                  f"reference slide(s), {len(ref_df)} nuclei, into the training pool "
                  f"({df['wsi_name'].nunique()} slides total before the random holdout split)")

    named_phase2 = set(args.phase2_slides) if args.phase2_slides else set()
    phase2_ann_dirs = set(args.ann_dir[1:]) if isinstance(args.ann_dir, list) and len(args.ann_dir) > 1 else set()
    df["_is_phase2"] = (
        (df["_phase_source"] == 2)
        | df["wsi_name"].isin(named_phase2)
        | df.get("ann_source_dir", pd.Series(dtype=object)).isin(phase2_ann_dirs)
    )
    n_p2_slides = df.loc[df["_is_phase2"], "wsi_name"].nunique()
    n_p1_slides = df.loc[~df["_is_phase2"], "wsi_name"].nunique()
    print(f"\nCombined: {n_p1_slides} phase-1 slides, {n_p2_slides} phase-2 slides "
          f"(weight x{args.phase2_weight} on phase 2)")
    if not args.phase2_feat_dir and not named_phase2:
        print("  (phase1/phase2 split not in use this run - all data treated as phase 1)")

    df["_label"] = df["label"].apply(cf.collapse_label)
    dropped = df[df["_label"].isna()]
    if len(dropped):
        print(f"\nDropped {len(dropped)} rows with unmapped/excluded raw labels:")
        for k, v in dropped["label"].value_counts().items():
            print(f"  '{k}': {v}")
    df = df.dropna(subset=["_label"])
    df = df[df["_label"].isin(args.classes)].reset_index(drop=True)
    moe.add_derived_features(df)  # e.g. "Derived: Aspect ratio" - only used if the feature set lists it

    discovered_cols = [c for c in df.columns
                       if c not in NON_FEATURE_COLS and not c.startswith("_")
                       and c not in ("subtype", "slide_uid")
                       and pd.api.types.is_numeric_dtype(df[c])
                       and df[c].notna().any()]

    if args.feature_list:
        wanted = load_feature_list_file(args.feature_list)
        source = f"--feature_list {args.feature_list}"
    else:
        wanted = cf.FEATURE_SETS[args.feature_set]
        source = f"--feature_set {args.feature_set!r}"
    feat_cols = [c for c in wanted if c in discovered_cols]
    missing = [c for c in wanted if c not in discovered_cols]
    print(f"\n{source}: {len(feat_cols)}/{len(wanted)} feature(s) matched in this data")
    if missing:
        print(f"  NOT found (dropped, not median-filled): {missing[:10]}"
              + (" ..." if len(missing) > 10 else ""))

    moe_groups = None
    if args.arch == "moe":
        if args.moe_drop_groups:
            # Remove ablated groups' columns from the input entirely (gate included),
            # so the ablation is a true "without these features" model.
            keep = set(i for idx in moe.assign_groups(feat_cols, args.moe_drop_groups).values()
                       for i in idx)
            dropped_cols = [c for i, c in enumerate(feat_cols) if i not in keep]
            feat_cols = [c for i, c in enumerate(feat_cols) if i in keep]
            print(f"  --moe_drop_groups {args.moe_drop_groups}: removed {len(dropped_cols)} feature(s)")
        moe_groups = moe.assign_groups(feat_cols)
        moe.print_groups(moe_groups, feat_cols)

    print(f"\n{len(df)} nuclei | {df['slide_uid'].nunique()} slides | {len(feat_cols)} features")
    print("\nClass counts:\n", df["_label"].value_counts())
    print("\nPooled training data by subtype:")
    print(df.groupby("subtype").agg(nuclei=("_label", "size"),
                                    slides=("wsi_name", "nunique"),
                                    tumor_frac=("_label", lambda s: (s == "Tumor").mean()))
          .round(3).to_string())
    missing_st = [s for s in args.subtypes if s not in set(df["subtype"])]
    if missing_st:
        print(f"  NOTE: no training slides found for {missing_st}")

    # Median imputation - stored in the checkpoint so inference fills gaps identically.
    medians = df[feat_cols].median()
    df[feat_cols] = df[feat_cols].fillna(medians)

    classes = list(args.classes)
    c2i = {c: i for i, c in enumerate(classes)}
    X_all = df[feat_cols].to_numpy(dtype=np.float64)
    y_all = df["_label"].map(c2i).to_numpy()

    n_slides = df["slide_uid"].nunique()
    use_holdout = args.val_source == "holdout" and args.holdout_frac > 0 and n_slides > 1
    if args.val_source == "holdout" and args.holdout_frac > 0 and n_slides <= 1:
        print(f"\nOnly {n_slides} slide(s) available - skipping holdout, training on all data.")

    if use_holdout:
        slides = sorted(df["slide_uid"].unique())
        rng = np.random.RandomState(args.seed)
        rng.shuffle(slides)
        n_hold = max(1, int(len(slides) * args.holdout_frac))
        n_hold = min(n_hold, len(slides) - 1)  # always keep at least one training slide
        hold = set(slides[:n_hold])
        te = df["slide_uid"].isin(hold).to_numpy()
        print(f"\nHolding out {n_hold} slides for sanity check: {sorted(hold)}")
    else:
        te = np.zeros(len(df), dtype=bool)
        if args.val_source == "test":
            print("\nTraining on ALL slides - validating against --test_feat_dir instead of a holdout split.")
        else:
            print("\nTraining on ALL slides (no holdout) - deployment model.")

    tr = ~te
    X_tr, y_tr = X_all[tr], y_all[tr]
    phase_tr = np.where(df["_is_phase2"].to_numpy()[tr], 2, 1)
    subtype_tr = df["subtype"].to_numpy()[tr]

    mu, sd = X_tr.mean(0), X_tr.std(0)
    sd[sd == 0] = 1.0
    X_tr_s = np.clip((X_tr - mu) / sd, -20, 20).astype(np.float32)

    # Keep the whole training set resident on `device` and sample weighted
    # batches there directly (small tabular MLP - the full set fits easily).
    X_tr_t = torch.from_numpy(X_tr_s).to(device)
    y_tr_t = torch.from_numpy(y_tr).to(device)
    counts = Counter(y_tr.tolist())
    w = np.array([1.0 / (counts[int(t)] ** args.reweight_power) for t in y_tr])
    if args.phase2_weight != 1.0:
        w = w * np.where(phase_tr == 2, args.phase2_weight, 1.0)
    if args.subtype_balance > 0:
        st_counts = Counter(subtype_tr.tolist())
        w = w * np.array([1.0 / (st_counts[s] ** args.subtype_balance) for s in subtype_tr])
        share = pd.Series(w).groupby(subtype_tr).sum() / w.sum()
        print("  sampling share per subtype after --subtype_balance "
              f"{args.subtype_balance}: " + ", ".join(f"{k}={v:.1%}" for k, v in share.items()))
    w_t = torch.as_tensor(w, dtype=torch.double, device=device)
    steps_per_epoch = max(1, len(y_tr) // args.batch_size)


    # Stain vocabulary for the MoE gate (index len(vocab) = "unknown").
    subtype_vocab = sorted(set(args.subtypes))
    st_tr_t = torch.as_tensor(moe.subtype_indices(subtype_tr, subtype_vocab), device=device)

    moe_config = None
    if args.arch == "moe":
        moe_config = {
            "groups": [(k, list(v)) for k, v in moe_groups.items()],
            "n_subtypes": len(subtype_vocab), "d": args.moe_dim,
            "expert_hidden": args.moe_expert_hidden, "gate_hidden": args.moe_gate_hidden,
            "head_hidden": args.moe_head_hidden, "dropout": args.moe_dropout,
            "subtype_dim": 8, "use_subtype": not args.moe_no_subtype_gate,
            "gate_temp": args.moe_gate_temp, "expert_dropout": args.moe_expert_dropout,
            "subtype_drop": args.moe_subtype_drop,
        }

    def make_model():
        if args.arch == "moe":
            return moe.build_from_config(moe_config, len(feat_cols), len(classes)).to(device)
        return MLP(len(feat_cols), len(classes), tuple(args.hidden), args.dropout).to(device)

    model = make_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {args.arch.upper()}  ({n_params:,} parameters)"
          + (f"  stain-conditioned gate over {subtype_vocab}" if moe_config and moe_config["use_subtype"] else ""))
    crit = nn.CrossEntropyLoss()

    # ---------------------------------------------------- domain-adversarial
    # TRAINING ONLY - uses only the training slides' own IDs, no unlabeled
    # data needed. The domain head's parameters are optimized normally; the
    # gradient flowing back INTO the shared trunk from it is reversed (see
    # grad_reverse/_GradReverseFn), so the trunk is pushed away from
    # encoding slide/site identity while the task loss pushes it toward
    # good Tumor/Non-Tumor separation. Never touches validation/test.
    domain_head = None
    if args.domain_adversarial:
        domain_key = (df.loc[tr, "wsi_name"].map(infer_site_from_slide)
                     if args.domain_level == "site" else df.loc[tr, "slide_uid"])
        domain_codes_tr, domain_levels = pd.factorize(domain_key)
        n_domains = len(domain_levels)
        domain_codes_tr_t = torch.as_tensor(domain_codes_tr, dtype=torch.long, device=device)
        domain_head = nn.Sequential(
            nn.Linear(model.feat_dim, args.domain_hidden), nn.ReLU(),
            nn.Linear(args.domain_hidden, n_domains)).to(device)
        # Domain classes are usually imbalanced (one site can dominate training
        # nuclei). Unweighted CE lets the head settle for "always predict the
        # majority domain" - a trivial fixed point that gives the reversed
        # gradient nothing real to fight, so the trunk never gets pushed
        # toward hiding domain identity. Class-weight it (inverse frequency)
        # so the head is forced to actually try to discriminate first.
        dom_counts = np.bincount(domain_codes_tr, minlength=n_domains).astype(np.float64)
        dom_weight = torch.as_tensor(dom_counts.sum() / (n_domains * dom_counts),
                                     dtype=torch.float32, device=device)
        domain_crit = nn.CrossEntropyLoss(weight=dom_weight)
        majority_baseline = dom_counts.max() / dom_counts.sum()
        print(f"  domain-adversarial: ON - predicting {args.domain_level} identity "
              f"({n_domains} classes: {sorted(domain_levels)}, counts "
              f"{dict(zip(domain_levels, dom_counts.astype(int)))}), "
              f"majority-class baseline={majority_baseline:.4f}, "
              f"lambda_max={args.domain_lambda_max}, domain_hidden={args.domain_hidden}")
        opt = torch.optim.Adam(list(model.parameters()) + list(domain_head.parameters()),
                               lr=args.lr, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Build validation tensors from whichever source --val_source picked.
    val_label = "holdout"
    sub_te = slide_te = None
    if args.val_source == "test":
        val_label = "test set"
        test_subtypes = args.test_subtypes or args.subtypes
        df_ext = load_eval_set(args, test_subtypes, "validation (test set)")
        if df_ext is None or df_ext.empty:
            raise SystemExit("--val_source test but no validation data could be loaded.")
        df_ext, X_te_s, y_te = load_and_label(df_ext, classes, feat_cols, medians, mu, sd)
        if df_ext.empty:
            raise SystemExit("--val_source test: no rows left after label filtering.")
        sub_te = df_ext["subtype"].to_numpy()
        slide_te = df_ext["slide_uid"].to_numpy()
        print(f"  validation (test set): {len(df_ext)} nuclei | {df_ext['slide_uid'].nunique()} slides")
        print(df_ext.groupby("subtype").agg(nuclei=("_label", "size"),
                                            slides=("wsi_name", "nunique"))
              .to_string())
        no_val = [s for s in sorted(set(df["subtype"])) if s not in set(sub_te)]
        if no_val:
            print(f"  NOTE: no validation slides for {no_val} - the pooled model is "
                  f"trained on them but not verified on them")
        slide_overlap_check(df.loc[tr, "wsi_name"], df_ext["wsi_name"])
        has_val = True
    else:
        has_val = te.any()
        X_te_s = np.clip((X_all[te] - mu) / sd, -20, 20).astype(np.float32) if has_val else None
        y_te = y_all[te] if has_val else None
        if has_val:
            sub_te = df["subtype"].to_numpy()[te]
            slide_te = df["slide_uid"].to_numpy()[te]

    X_te_t = torch.from_numpy(X_te_s).to(device) if has_val else None
    y_te_t = torch.from_numpy(y_te).to(device) if has_val else None
    st_te_t = (torch.as_tensor(moe.subtype_indices(sub_te, subtype_vocab), device=device)
               if has_val else None)
    # Index tensors per validation subtype (for per-subtype loss/acc every eval).
    val_groups = ({st: torch.as_tensor(np.flatnonzero(sub_te == st), device=device)
                   for st in sorted(set(sub_te))} if has_val else {})
    if has_val and args.val_select == "macro" and len(val_groups) > 1:
        print(f"  best checkpoint / early stop / plateau use the MACRO val loss "
              f"(mean over {sorted(val_groups)})")

    if args.early_stop and not has_val:
        print("  NOTE: --early_stop ignored - no validation set available.")
    if args.lr_schedule == "plateau" and not has_val:
        print("  NOTE: --lr_schedule plateau ignored - no validation set available.")

    scheduler = None
    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    elif args.lr_schedule == "plateau" and has_val:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=max(args.patience // 2, 1))

    def ckpt_dict(epoch, train_metrics, val_metrics):
        d = {
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "epoch": epoch,
            "classes": classes,
            "qupath_names": [QUPATH_NAMES.get(c, c) for c in classes],
            "feature_cols": feat_cols,
            "mu": mu, "sd": sd,
            "medians": medians[feat_cols].to_numpy(),
            "hidden": args.hidden, "dropout": args.dropout,
            "arch": args.arch,
            "moe_config": moe_config,
            "subtype_vocab": subtype_vocab,
            "subtypes": args.subtypes,
            "subtype_balance": args.subtype_balance,
            "domain_adversarial": args.domain_adversarial,
            "domain_level": args.domain_level if args.domain_adversarial else None,
            "domain_lambda_max": args.domain_lambda_max if args.domain_adversarial else None,
            "domain_acc_final": domain_acc_ema,
            "feature_set": args.feature_set if not args.feature_list else None,
            "feature_list_file": args.feature_list,
            "n_train_nuclei": int(tr.sum()),
            "n_train_slides": int(df.loc[tr, "slide_uid"].nunique()),
            "n_train_nuclei_by_subtype": {k: int(v) for k, v in Counter(subtype_tr.tolist()).items()},
            "phase2_slides": sorted(df.loc[df["_is_phase2"], "wsi_name"].unique().tolist()),
            "phase2_weight": args.phase2_weight,
            "train_loss": train_metrics.loss, "train_acc": train_metrics.acc,
        }
        if val_metrics.epoch >= 0:
            d["val_loss"], d["val_acc"] = val_metrics.loss, val_metrics.acc
            d["val_select"] = args.val_select
            d["val_by_subtype"] = dict(last_val_by_subtype)
        return d

    ckpt_best_path = args.ckpt_best or f"{args.out}.best"

    writer = None
    if not args.no_tb:
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = args.tb_dir or f"{os.path.splitext(args.out)[0]}_tb"
        writer = SummaryWriter(log_dir=tb_dir)
        print(f"TensorBoard logs -> {tb_dir}  (tensorboard --logdir {tb_dir})")

    train_metrics = Metrics("train")
    val_metrics = Metrics("val")
    since_best_loss = 0
    stopped_early = False
    last_epoch = args.epochs
    best_val_acc_at_best_loss = None
    last_val_by_subtype = {}

    domain_acc_ema = None
    for ep in range(1, args.epochs + 1):
        last_epoch = ep
        moe_parts_acc = []
        model.train()
        if domain_head is not None:
            domain_head.train()
            lambd = domain_lambda_schedule((ep - 1) / max(args.epochs - 1, 1),
                                           args.domain_lambda_max)
        for _ in range(steps_per_epoch):
            idx = torch.multinomial(w_t, args.batch_size, replacement=True)
            xb, yb, stb = X_tr_t[idx], y_tr_t[idx], st_tr_t[idx]
            opt.zero_grad()
            feats = None
            if args.arch == "moe":
                outputs, aux = model(xb, stb, return_aux=True)
                feats = aux["fused"]
                extra, parts = moe.auxiliary_loss(aux, yb, args.moe_aux_weight,
                                                  args.moe_balance_weight, args.moe_bag)
                loss = crit(outputs, yb) + extra
                moe_parts_acc.append(parts)
            elif domain_head is not None:
                feats = model.features(xb)
                outputs = model.classify(feats)
                loss = crit(outputs, yb)
            else:
                outputs = model(xb)
                loss = crit(outputs, yb)
            if domain_head is not None:
                dom_logits = domain_head(grad_reverse(feats, lambd))
                dom_target = domain_codes_tr_t[idx]
                loss = loss + domain_crit(dom_logits, dom_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            if domain_head is not None:
                torch.nn.utils.clip_grad_norm_(domain_head.parameters(), max_norm=5.0)
            opt.step()

        # Full-pass train metrics (representative of deployment performance,
        # unlike the resampled/balanced batches used for training).
        recompute = ep % max(args.val_every, 1) == 0 or ep == 1 or ep == args.epochs
        if recompute:
            model.eval()
            with torch.no_grad():
                tr_logits = model(X_tr_t, st_tr_t)
                tr_pred_full = tr_logits.argmax(1).cpu().numpy()
                train_loss = crit(tr_logits, y_tr_t).item()
            train_pc_acc, _ = per_class_accuracy(y_tr, tr_pred_full, classes)
            train_acc = float(np.mean(tr_pred_full == y_tr))
            train_improved = train_metrics.update(ep, train_loss, train_acc, train_pc_acc)
            if writer is not None:
                train_metrics.to_writer(writer)
            if domain_head is not None:
                domain_head.eval()
                with torch.no_grad():
                    dom_pred = domain_head(model.features(X_tr_t, st_tr_t)).argmax(1).cpu().numpy()
                dom_pc_acc, _ = per_class_accuracy(domain_codes_tr, dom_pred,
                                                   list(range(n_domains)))
                domain_acc_ema = float(np.mean(list(dom_pc_acc.values())))  # macro/balanced
                if writer is not None:
                    writer.add_scalar("domain/balanced_acc", domain_acc_ema, ep)
                    writer.add_scalar("domain/lambda", lambd, ep)
        else:
            train_improved = False

        do_eval = has_val and (ep % args.val_every == 0 or ep == args.epochs)
        val_loss_improved = False
        if do_eval:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_te_t, st_te_t)
                pooled_loss = crit(val_logits, y_te_t).item()
                val_pred_t = val_logits.argmax(1)
                last_val_by_subtype = {
                    st: (crit(val_logits[g], y_te_t[g]).item(),
                         (val_pred_t[g] == y_te_t[g]).float().mean().item())
                    for st, g in val_groups.items()}
                val_pred = val_pred_t.cpu().numpy()
            val_acc = float(np.mean(val_pred == y_te))
            if args.val_select == "macro" and last_val_by_subtype:
                val_loss = float(np.mean([l for l, _ in last_val_by_subtype.values()]))
            else:
                val_loss = pooled_loss
            val_pc_acc, _ = per_class_accuracy(y_te, val_pred, classes)
            val_metrics.update(ep, val_loss, val_acc, val_pc_acc)
            if writer is not None:
                val_metrics.to_writer(writer)
                writer.add_scalar("val/loss_pooled", pooled_loss, ep)
                for st, (l, a) in last_val_by_subtype.items():
                    writer.add_scalar(f"val_{st}/loss", l, ep)
                    writer.add_scalar(f"val_{st}/acc", a, ep)

            val_loss_improved = val_metrics.min_loss_epoch == ep
            since_best_loss = 0 if val_loss_improved else since_best_loss + 1

            if scheduler is not None and args.lr_schedule == "plateau":
                scheduler.step(val_loss)

        if scheduler is not None and args.lr_schedule == "cosine":
            scheduler.step()

        moe_usage = None
        if moe_parts_acc:
            moe_usage = np.mean([p["usage"] for p in moe_parts_acc], axis=0)
            if writer is not None:
                writer.add_scalar("moe/aux_ce", np.mean([p["aux_ce"] for p in moe_parts_acc]), ep)
                writer.add_scalar("moe/balance", np.mean([p["balance"] for p in moe_parts_acc]), ep)
                for gname, u in zip(model.group_names, moe_usage):
                    writer.add_scalar(f"moe_gate/{gname}", float(u), ep)

        # Checkpoint on improvement: val loss when a val set exists, else train acc.
        improved = val_loss_improved if has_val else train_improved
        if improved:
            torch.save(ckpt_dict(ep, train_metrics, val_metrics), ckpt_best_path)
            if has_val:
                best_val_acc_at_best_loss = val_acc

        if ep % 10 == 0 or ep == 1 or ep == args.epochs:
            msg = f"  epoch {ep:03d}  {train_metrics.to_str()}"
            if moe_usage is not None:
                msg += "\n              gate usage: " + "  ".join(
                    f"{n}={u:.2f}" for n, u in zip(model.group_names, moe_usage))
            if domain_head is not None and domain_acc_ema is not None:
                chance = 1.0 / n_domains  # the right comparison for a BALANCED metric
                msg += (f"\n              domain_balanced_acc={domain_acc_ema:.4f} "
                       f"(chance={chance:.4f}, lambda={lambd:.3f}) - lower is better: the "
                       f"trunk is hiding {args.domain_level} identity. (Class-weighted, "
                       f"macro-recall - not the raw accuracy a {majority_baseline:.4f} "
                       f"majority-class baseline could otherwise trivially reach.)")
            if do_eval:
                msg += f"\n              {val_metrics.to_str()}"
                if len(last_val_by_subtype) > 1:
                    msg += "\n              by subtype: " + "  ".join(
                        f"{st} loss={l:.4f} acc={a:.4f}"
                        for st, (l, a) in last_val_by_subtype.items())
                if scheduler is not None:
                    msg += f"\n              lr={opt.param_groups[0]['lr']:.2e}"
                if args.early_stop:
                    msg += f"  (since_best_loss={since_best_loss}/{args.patience})"
            print(msg)

        if args.early_stop and has_val and do_eval and since_best_loss >= args.patience:
            print(f"  early stopping at epoch {ep} - no val-loss improvement for "
                  f"{args.patience} evaluations (best was epoch {val_metrics.min_loss_epoch}, "
                  f"val_loss {val_metrics.min_loss:.4f})")
            stopped_early = True
            break

    if writer is not None:
        writer.close()

    if args.early_stop and has_val:
        best_ckpt = torch.load(ckpt_best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state"])
        print(f"  reloaded best-epoch weights (epoch {best_ckpt['epoch']}) for the final save"
              + (" after early stopping" if stopped_early else ""))

    print(f"epoch {last_epoch} lr={opt.param_groups[0]['lr']:.6f}")

    if has_val:
        model.eval()
        with torch.no_grad():
            pred = model(X_te_t, st_te_t).argmax(1).cpu().numpy()
        tag = "best epoch" if args.early_stop else "final epoch"
        if args.early_stop:
            # `model` already IS the reloaded best checkpoint - full report.
            report_eval_by_subtype(f"VALIDATION ({val_label.upper()}, {tag})",
                                   y_te, pred, classes, sub_te, slide_te)
        else:
            # Brief one-liner here; the full report below is for the
            # checkpoint that's actually deployed (.best), which is what
            # matters - printing both in full just duplicates the same
            # numbers twice when training didn't diverge.
            from sklearn.metrics import f1_score
            print(f"\n{val_label.upper()} ({tag}): acc={np.mean(pred == y_te):.4f}  "
                  f"macro_f1={f1_score(y_te, pred, average='macro', zero_division=0):.4f}")
        acc_str = f"{best_val_acc_at_best_loss:.4f}" if best_val_acc_at_best_loss is not None else "n/a"
        print(f"Best val loss ({args.val_select}): {val_metrics.min_loss:.4f} @ epoch "
              f"{val_metrics.min_loss_epoch} (val acc {acc_str} at that epoch) -> saved to "
              f"{ckpt_best_path}")

        # Full report for the checkpoint that's actually deployed (.best).
        if not args.early_stop and os.path.exists(ckpt_best_path):
            best_ckpt = torch.load(ckpt_best_path, map_location=device, weights_only=False)
            best_model = make_model()
            best_model.load_state_dict(best_ckpt["model_state"])
            best_model.eval()
            with torch.no_grad():
                best_pred = best_model(X_te_t, st_te_t).argmax(1).cpu().numpy()
            report_eval_by_subtype(
                f"VALIDATION ({val_label.upper()}, best checkpoint epoch {best_ckpt['epoch']})",
                y_te, best_pred, classes, sub_te, slide_te)
            if args.arch == "moe":
                moe.report_experts(best_model, X_te_t, st_te_t, y_te, sub_te, classes,
                                   f"{val_label}, best checkpoint")
        if args.arch == "moe" and args.early_stop:
            moe.report_experts(model, X_te_t, st_te_t, y_te, sub_te, classes,
                               f"{val_label}, best epoch")

    if args.val_source != "test" and args.pool_reference_into_train:
        print("\n(EXTERNAL TEST SET skipped: --pool_reference_into_train means these "
              "reference slides are likely already in the training pool - scoring them "
              "again here would not be a valid test. See the VALIDATION (HOLDOUT, ...) "
              "report above instead.)")
    elif args.val_source != "test":
        print("\n=== EXTERNAL TEST SET ===")
        test_subtypes = args.test_subtypes or args.subtypes
        df_ext = load_eval_set(args, test_subtypes, "external test")
        if df_ext is None:
            print("  no external test data could be loaded - skipping")
        elif "label" not in df_ext.columns:
            print("  --test_feat_dir data has no 'label' column - skipping")
        else:
            df_ext, X_ext_s, y_ext = load_and_label(df_ext, classes, feat_cols, medians, mu, sd)
            if df_ext.empty:
                print("  no rows left after label filtering - skipping")
            else:
                slide_overlap_check(df.loc[tr, "wsi_name"], df_ext["wsi_name"], "external test")
                model.eval()
                with torch.no_grad():
                    st_ext_t = torch.as_tensor(moe.subtype_indices(
                        df_ext["subtype"].to_numpy(), subtype_vocab), device=device)
                    ext_pred = model(torch.from_numpy(X_ext_s).to(device),
                                     st_ext_t).argmax(1).cpu().numpy()
                print(f"  {len(df_ext)} nuclei | {df_ext['slide_uid'].nunique()} slides")
                report_eval_by_subtype("EXTERNAL TEST", y_ext, ext_pred, classes,
                                       df_ext["subtype"].to_numpy(),
                                       df_ext["slide_uid"].to_numpy())

    torch.save(ckpt_dict(last_epoch, train_metrics, val_metrics), args.out)
    if args.early_stop and has_val:
        print(f"\nSaved best-epoch model (epoch {val_metrics.min_loss_epoch}) -> {args.out}")
    else:
        print(f"\nSaved final-epoch model (epoch {last_epoch}) -> {args.out}")
    if not has_val:
        print(f"Saved best-train-acc checkpoint -> {ckpt_best_path} "
              f"(epoch {train_metrics.max_acc_epoch}, acc {train_metrics.max_acc:.4f})")
    print(f"  {len(feat_cols)} features, classes: {classes}")
    print(f"  QuPath names: {[QUPATH_NAMES.get(c, c) for c in classes]}")


if __name__ == "__main__":
    main()
