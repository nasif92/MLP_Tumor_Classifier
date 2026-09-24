"""
Trains ONE pooled model on all five subtypes and saves it for deployment.

Usage:
    python train_deploy.py --feat_dir . --ann_dir . --out pooled_model.pt
    python train_deploy.py --feat_dir . --ann_dir . --holdout_frac 0.1   # sanity check
    # Train on phase 1 only, evaluate on phase 2 as a fully separate test set:
    python train_deploy.py --feat_dir <phase1_feat_dir> --ann_dir <phase1_ann_dir> \\
        --phase2_feat_dir <phase2_feat_dir> --phase2_ann_dir <phase2_ann_dir> \\
        --holdout_frac 0 --out phase1_plus_phase2.pt

    # Single-slide training (e.g. one small dataset):
    python3 train_deploy.py --feat_dir <dir_with_one_slide> --ann_dir <ann_dir> --out one_slide.pt
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

# Curated, pre-classified reference/validation slides - used as the default
# --test_feat_dir so a plain run evaluates against this set automatically.
REFERENCE_TEST_DIR = cf.ref_set_ki67_all_detections
REFERENCE_ANN_DIR = cf.ref_set_ann_ki67
# Single unified annotation directory covering ALL reference slides/subtypes
# (previously fragmented per-subtype, e.g. "annotations-rds", "annotations-cis" -
# now consolidated so --ann_dir/--test_ann_dir need only ever point here).

NON_FEATURE_COLS = {
    "wsi_name", "nucleus_id", "cx_wsi", "cy_wsi", "label", "is_ground_truth",
    "containing_annotation_ids", "containing_annotation_classes",
    "annotation_nesting_depth",
}


def _infer_subtype_from_name(slide_name, subtypes):
    names = f"_{slide_name}_".lower()
    for st in subtypes:
        if f"_{st}_".lower() in names:
            return st
    return None


def _load_geojson_any(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    feats = data["features"] if isinstance(data, dict) and "features" in data else (
        data if isinstance(data, list) else []
    )
    # Some exports mix annotation-type objects into the detection/feature
    # file alongside real nuclei. Only real detections have genuine
    # per-nucleus measurements - filter strictly so an annotation shape
    # can never masquerade as a nucleus downstream.
    return [f for f in feats
            if (f.get("properties", {}) or {}).get("objectType") == "detection"]


def _centroid_of_ring(ring):
    return np.asarray(ring, dtype=float).mean(axis=0)


def _polygon_centroid(geometry):
    """Centroid of a GeoJSON geometry dict (Point / Polygon / MultiPolygon)."""
    gtype = geometry.get("type")
    coords = geometry["coordinates"]
    if gtype == "Point":
        return np.asarray(coords, dtype=float)
    if gtype == "Polygon":
        return _centroid_of_ring(coords[0])
    if gtype == "MultiPolygon":
        rings = [poly[0] for poly in coords if poly]
        pts = [_centroid_of_ring(r) for r in rings if r]
        return np.mean(pts, axis=0) if pts else np.array([np.nan, np.nan])
    return np.array([np.nan, np.nan])


def _find_annotation_file(ann_dir, slide_name):
    """Recursively search ann_dir for this slide's annotations file.
    ann_dir may be a single directory or a list of directories - searched
    in order, first match wins (lets --ann_dir combine multiple sources,
    e.g. phase1_annotations + phase2_annotations, in one training run)."""
    candidates = [f"{slide_name}.geojson", f"{slide_name}.geojson.gz", 
    f"{slide_name}_annotations.geojson", f"{slide_name}_annotations.geojson.gz"]    
    
    ann_dirs = [ann_dir] if isinstance(ann_dir, str) else list(ann_dir)
    for d in ann_dirs:
        for dirpath, _, filenames in os.walk(d):
            for fn in filenames:
                if fn in candidates:
                    return os.path.join(dirpath, fn), d
    return None, None


def _labels_from_classification(cell_features):
    """Use each detection's own classification as its label."""
    labels = []
    for f in cell_features:
        cls = f["properties"].get("classification")
        name = cls.get("name") if isinstance(cls, dict) else None
        labels.append(name if name and name.lower() != "nucleus" else None)
    return labels


def _load_slide_features_and_labels(feat_path, ann_path, slide_name):
    cell_features = _load_geojson_any(feat_path)
    if not cell_features:
        return None

    centroids = np.array([_polygon_centroid(f["geometry"]) for f in cell_features])

    if ann_path is not None:
        import extract_features as ef

        # Reject any detection whose centroid falls outside the union
        # bbox of all annotation regions BEFORE building measurement
        # arrays for it - on a whole-slide feature file this is the
        # difference between processing hundreds of thousands of
        # detections and processing only the few thousand actually
        # inside annotated regions.
        ann_bbox = ef.annotation_bbox(ann_path)
        if ann_bbox is not None:
            x0, x1, y0, y1 = ann_bbox
            in_bbox = ((centroids[:, 0] >= x0) & (centroids[:, 0] <= x1) &
                       (centroids[:, 1] >= y0) & (centroids[:, 1] <= y1))
            n_before = len(cell_features)
            cell_features = [f for f, keep in zip(cell_features, in_bbox) if keep]
            centroids = centroids[in_bbox]
            print(f"    {slide_name}: {n_before} detections -> "
                  f"{len(cell_features)} inside annotation bbox")
            if not cell_features:
                return None

        labels = ef.assign_labels(centroids, ann_path, verbose=False)
    else:
        labels = _labels_from_classification(cell_features)

    keep = [i for i, label in enumerate(labels) if label is not None]
    cell_features = [cell_features[i] for i in keep]
    centroids = centroids[keep]
    labels = [labels[i] for i in keep]
    if not cell_features:
        return None

    n = len(cell_features)
    all_names = sorted({k for f in cell_features
                         for k in f["properties"].get("measurements", {})})

    # Columnar construction (fast) instead of a list of per-nucleus dicts.
    m_arrays = {name: np.full(n, np.nan, dtype=np.float64) for name in all_names}
    for i, f in enumerate(cell_features):
        for name, val in f["properties"].get("measurements", {}).items():
            if val is not None:
                m_arrays[name][i] = val

    data = {
        "wsi_name": slide_name,
        "nucleus_id": np.arange(n),
        "cx_wsi": centroids[:, 0].astype(np.float64),
        "cy_wsi": centroids[:, 1].astype(np.float64),
        "label": labels,
        "is_ground_truth": [l is not None for l in labels],
    }
    data.update(m_arrays)
    return pd.DataFrame(data)


def _load_geojson_root(feat_dir, ann_dir, subtypes, feat_pattern=".geojson.gz",
                        label="data", use_classification=False):
    if not feat_dir:
        return None
    frames = []
    for dirpath, _, filenames in os.walk(feat_dir):
        for fn in filenames:
            if not fn.endswith(feat_pattern):
                continue
            slide_name = fn[: -len(feat_pattern)]
            st = _infer_subtype_from_name(slide_name, subtypes)
            if st is None:
                continue
            if use_classification:
                ann_path = None
                matched_ann_dir = None
            else:
                ann_path, matched_ann_dir = _find_annotation_file(ann_dir, slide_name) if ann_dir else (None, None)

                if ann_path is None:
                    print(f"  {slide_name}: no annotations found under {ann_dir} - skipping")
                    continue
            try:
                df_slide = _load_slide_features_and_labels(
                    os.path.join(dirpath, fn), ann_path, slide_name)
            except Exception as e:
                print(f"  {slide_name}: FAILED to load {fn} ({type(e).__name__}: {e}) - skipping")
                continue
            if df_slide is None or df_slide.empty:
                continue
            df_slide = pd.concat([df_slide, pd.DataFrame({
                "subtype": st,
                "slide_uid": st + ":" + df_slide["wsi_name"].astype(str),
                "ann_source_dir": matched_ann_dir,
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
                     feat_pattern, str(use_classification)])
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

    def forward(self, x):
        return self.net(x)


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
    X = df_ext.reindex(columns=feat_cols).astype(np.float64)
    X = X.fillna(medians[feat_cols])
    X_s = ((X.to_numpy() - mu) / sd).astype(np.float32)
    c2i = {c: i for i, c in enumerate(classes)}
    y = df_ext["_label"].map(c2i).to_numpy()
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
    ap.add_argument("--subtypes", nargs="+", default=cf.DEFAULT_SUBTYPES)
    ap.add_argument("--classes", nargs="+", default=["Tumor", "Non Tumor"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout_frac", type=float, default=0.0,
                    help="Hold out this fraction of SLIDES for a sanity check. "
                         "Use 0 for the final deployment model. Ignored when "
                         "--val_source test, or when only one slide is available.")
    ap.add_argument("--val_source", choices=["holdout", "test"], default="test",
                    help="Where periodic validation comes from. 'holdout' carves out "
                         "--holdout_frac of the training slides. 'test' (default) uses "
                         "--test_feat_dir/--test_ann_dir instead, training on ALL slides.")
    ap.add_argument("--hidden", nargs="+", type=int, default=[96, 48, 24, 12, 6])
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=10000)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--reweight_power", type=float, default=0.5)
    ap.add_argument("--phase2_slides", nargs="+", default=None,
                    help="Slide names (wsi_name) to up-weight as phase 2 during retraining, "
                         "even if loaded from --feat_dir.")
    ap.add_argument("--phase2_weight", type=float, default=1.5,
                    help="Sampling-weight multiplier for phase-2 slides. Only has any "
                         "effect when --phase2_feat_dir or --phase2_slides is actually "
                         "given - kept at a ready-to-use default (1.5x) for whenever "
                         "phase-2 data is reintroduced.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default='cuda',
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
    ap.add_argument("--test_feat_dir", default=REFERENCE_TEST_DIR,
                    help="Feature directory of held-out data to evaluate the trained model on. "
                         "NEVER used for training. Defaults to the full curated reference "
                         "slide set. Evaluation reuses this run's feature columns/medians/"
                         "mu/sd - never recomputes them from the test data.")
    ap.add_argument("--test_ann_dir", default="",
                    help="Annotation directory matching --test_feat_dir (the same unified "
                         "reference annotations directory). Pass an empty string to instead "
                         "use each nucleus's own pre-existing classification directly, with "
                         "no annotation join.")
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

    feat_cols = [c for c in df.columns
                 if c not in NON_FEATURE_COLS and not c.startswith("_")
                 and c not in ("subtype", "slide_uid")
                 and pd.api.types.is_numeric_dtype(df[c])
                 and df[c].notna().any()]
    print(f"\n{len(df)} nuclei | {df['slide_uid'].nunique()} slides | {len(feat_cols)} features")
    print("\nClass counts:\n", df["_label"].value_counts())

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
    w_t = torch.as_tensor(w, dtype=torch.double, device=device)
    steps_per_epoch = max(1, len(y_tr) // args.batch_size)

    model = MLP(len(feat_cols), len(classes), tuple(args.hidden), args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss()

    # Build validation tensors from whichever source --val_source picked.
    val_label = "holdout"
    if args.val_source == "test":
        val_label = "test set"
        test_subtypes = args.test_subtypes or args.subtypes
        df_ext = _load_geojson_root_cached(args.test_feat_dir, args.test_ann_dir, test_subtypes,
                                            cache_dir=args.cache_dir,
                                            feat_pattern=args.test_feat_pattern,
                                            label="validation (test set)",
                                            use_classification=not bool(args.test_ann_dir))
        if df_ext is None or df_ext.empty:
            raise SystemExit("--val_source test but no data found under --test_feat_dir/--test_ann_dir.")
        df_ext, X_te_s, y_te = load_and_label(df_ext, classes, feat_cols, medians, mu, sd)
        if df_ext.empty:
            raise SystemExit("--val_source test: no rows left after label filtering.")
        print(f"  validation (test set): {len(df_ext)} nuclei | {df_ext['slide_uid'].nunique()} slides")
        has_val = True
    else:
        has_val = te.any()
        X_te_s = ((X_all[te] - mu) / sd).astype(np.float32) if has_val else None
        y_te = y_all[te] if has_val else None

    X_te_t = torch.from_numpy(X_te_s).to(device) if has_val else None
    y_te_t = torch.from_numpy(y_te).to(device) if has_val else None

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
            "subtypes": args.subtypes,
            "n_train_nuclei": int(tr.sum()),
            "n_train_slides": int(df.loc[tr, "slide_uid"].nunique()),
            "phase2_slides": sorted(df.loc[df["_is_phase2"], "wsi_name"].unique().tolist()),
            "phase2_weight": args.phase2_weight,
            "train_loss": train_metrics.loss, "train_acc": train_metrics.acc,
        }
        if val_metrics.epoch >= 0:
            d["val_loss"], d["val_acc"] = val_metrics.loss, val_metrics.acc
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

    for ep in range(1, args.epochs + 1):
        last_epoch = ep
        model.train()
        for _ in range(steps_per_epoch):
            idx = torch.multinomial(w_t, args.batch_size, replacement=True)
            xb, yb = X_tr_t[idx], y_tr_t[idx]
            opt.zero_grad()
            outputs = model(xb)
            loss = crit(outputs, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

        # Full-pass train metrics (representative of deployment performance,
        # unlike the resampled/balanced batches used for training).
        recompute = ep % max(args.val_every, 1) == 0 or ep == 1 or ep == args.epochs
        if recompute:
            model.eval()
            with torch.no_grad():
                tr_pred_full = model(X_tr_t).argmax(1).cpu().numpy()
                train_loss = crit(model(X_tr_t), y_tr_t).item()
            train_pc_acc, _ = per_class_accuracy(y_tr, tr_pred_full, classes)
            train_acc = float(np.mean(tr_pred_full == y_tr))
            train_improved = train_metrics.update(ep, train_loss, train_acc, train_pc_acc)
            if writer is not None:
                train_metrics.to_writer(writer)
        else:
            train_improved = False

        do_eval = has_val and (ep % args.val_every == 0 or ep == args.epochs)
        val_loss_improved = False
        if do_eval:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_te_t)
                val_loss = crit(val_logits, y_te_t).item()
                val_pred = val_logits.argmax(1).cpu().numpy()
            val_acc = float(np.mean(val_pred == y_te))
            val_pc_acc, _ = per_class_accuracy(y_te, val_pred, classes)
            val_metrics.update(ep, val_loss, val_acc, val_pc_acc)
            if writer is not None:
                val_metrics.to_writer(writer)

            val_loss_improved = val_metrics.min_loss_epoch == ep
            since_best_loss = 0 if val_loss_improved else since_best_loss + 1

            if scheduler is not None and args.lr_schedule == "plateau":
                scheduler.step(val_loss)

        if scheduler is not None and args.lr_schedule == "cosine":
            scheduler.step()

        # Checkpoint on improvement: val loss when a val set exists, else train acc.
        improved = val_loss_improved if has_val else train_improved
        if improved:
            torch.save(ckpt_dict(ep, train_metrics, val_metrics), ckpt_best_path)
            if has_val:
                best_val_acc_at_best_loss = val_acc

        if ep % 10 == 0 or ep == 1 or ep == args.epochs:
            msg = f"  epoch {ep:03d}  {train_metrics.to_str()}"
            if do_eval:
                msg += f"\n              {val_metrics.to_str()}"
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
            pred = model(X_te_t).argmax(1).cpu().numpy()
        report_eval(f"VALIDATION ({val_label.upper()}, final epoch)", y_te, pred, classes)
        acc_str = f"{best_val_acc_at_best_loss:.4f}" if best_val_acc_at_best_loss is not None else "n/a"
        print(f"Best val loss: {val_metrics.min_loss:.4f} @ epoch {val_metrics.min_loss_epoch} "
              f"(val acc {acc_str} at that epoch) -> saved to {ckpt_best_path}")

    if args.test_feat_dir and args.val_source != "test":
        print(f"\n=== EXTERNAL TEST SET ({args.test_feat_dir}) ===")
        test_subtypes = args.test_subtypes or args.subtypes
        df_ext = _load_geojson_root_cached(args.test_feat_dir, args.test_ann_dir, test_subtypes,
                                            cache_dir=args.cache_dir,
                                            feat_pattern=args.test_feat_pattern,
                                            label="external test",
                                            use_classification=not bool(args.test_ann_dir))
        if df_ext is None:
            print("  no data found under --test_feat_dir/--test_ann_dir - skipping")
        elif "label" not in df_ext.columns:
            print("  --test_feat_dir data has no 'label' column - skipping")
        else:
            df_ext, X_ext_s, y_ext = load_and_label(df_ext, classes, feat_cols, medians, mu, sd)
            if df_ext.empty:
                print("  no rows left after label filtering - skipping")
            else:
                train_ids = set(zip(df.loc[tr, "wsi_name"], df.loc[tr, "nucleus_id"]))
                test_ids = set(zip(df_ext["wsi_name"], df_ext["nucleus_id"]))
                overlap = train_ids & test_ids
                print(f"  {len(overlap)} / {len(test_ids)} eval nuclei were also "
                    f"in the training set ({len(overlap)/len(test_ids):.2%})")
                model.eval()
                with torch.no_grad():
                    ext_pred = model(torch.from_numpy(X_ext_s).to(device)).argmax(1).cpu().numpy()
                print(f"  {len(df_ext)} nuclei | {df_ext['slide_uid'].nunique()} slides")
                report_eval("EXTERNAL TEST", y_ext, ext_pred, classes)

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
