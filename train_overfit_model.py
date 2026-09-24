"""
train_overfit_per_slide.py

For EACH slide under --feat_dir/--ann_dir, INDEPENDENTLY:
  1. Load every detection in the slide and run the annotation join
     (extract_features.assign_labels) to get each nucleus's true label,
     or None if it falls outside every annotated region.
  2. Fine-tune ("overfit") a FRESH copy of the pooled checkpoint's weights
     using ONLY the annotated nuclei from this one slide. Nothing carries
     over between slides - every slide starts from the same original
     pooled weights.
  3. Run BOTH the original (never fine-tuned) pooled model and this
     slide's freshly fine-tuned model on the UNANNOTATED nuclei only -
     the region the fine-tuning never saw - and export geojson comparing
     their calls there.

This answers: "if I let the model specialize on what's hand-annotated on
this one slide, does it generalize any better to the REST of that same
slide than the original pooled model does?"

Per slide, writes to <export_geojson_dir>/<subtype>/:
  <slide>_pooled_vs_overfit_unannotated.geojson - 4-class match/mismatch
      (overfit model's call vs pooled model's call), on unannotated nuclei
  <slide>_pooled_unannotated.geojson  - pooled model's raw call there
  <slide>_overfit_unannotated.geojson - this slide's overfit model's raw call there

Does NOT touch train_deploy.py, extract_features.py, or export_geojson.py -
imports them as libraries only.

Usage:
    python train_overfit_per_slide.py \\
        --pooled_model PR_model.pt \\
        --feat_dir <dir of whole-slide detection files> \\
        --ann_dir <matching annotation dir> \\
        --export_geojson_dir "Per-slide overfit classifications"

    # restrict to specific slides:
    python train_overfit_per_slide.py --pooled_model PR_model.pt \\
        --feat_dir ... --ann_dir ... --slides RDS24-012921_PR_4 CIS24-000938_PR_0 \\
        --export_geojson_dir "Per-slide overfit classifications"
"""
import argparse
import copy
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import config as cf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_deploy as td  # noqa: E402  - reused as a library only, never modified
import extract_features as ef  # noqa: E402
import export_geojson as eg  # noqa: E402  - reused for coloring/labeling conventions


def make_filtered_annotation_file(ann_path, exclude_names):
    """Writes a TEMP COPY of ann_path with any region whose classification
    name matches --exclude_annotation_names removed (case-insensitive) -
    e.g. a big umbrella "Main"/"Tissue" region that contains everything
    else. assign_labels() then only sees the real sub-annotations
    (Tumor, Stroma, Immune cells, ...), so a nucleus only gets a real
    label when it falls inside one of those - not the container region -
    and anything only covered by the excluded region correctly counts as
    unannotated. The original annotation file is never modified; the temp
    file is deleted by the caller when done."""
    if not exclude_names:
        return ann_path, None
    exclude_lower = {n.strip().lower() for n in exclude_names}
    feats = ef.load_geojson(ann_path)
    kept = []
    n_excluded = 0
    for f in feats:
        props = f.get("properties", {}) or {}
        cls = props.get("classification")
        nm = cls.get("name") if isinstance(cls, dict) else props.get("name")
        if nm and nm.strip().lower() in exclude_lower:
            n_excluded += 1
            continue
        kept.append(f)
    if n_excluded == 0:
        return ann_path, None  # nothing matched, no need for a temp file
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".geojson", delete=False)
    json.dump({"type": "FeatureCollection", "features": kept}, tmp)
    tmp.close()
    print(f"    excluded {n_excluded} region(s) matching {sorted(exclude_lower)} "
          f"from annotation join for this slide")
    return tmp.name, tmp.name


def build_slide_dataframe(feat_path, ann_path, slide_name, subtype):
    """One row per detection in the WHOLE slide (annotated or not).
    'label' is the raw annotation-join label, or None where the nucleus
    falls outside every annotated region. 'own_classification' is the
    nucleus's OWN pre-existing classification as already stored in the
    feature file (e.g. from an earlier RF classifier or pathologist call)
    - independent of the annotation join, this is what lets Ghost nuclei
    be identified and excluded from model inference entirely (see
    GHOST_EXCLUDE_NAMES below), not just from training.
    Returns (df, cell_features) where cell_features[i]['geometry'] is
    nucleus i's real polygon - df['nucleus_id'] == i always, by
    construction, so no separate reindexing/matching pass is ever needed
    to recover geometry later."""
    cell_features = td._load_geojson_any(feat_path)
    if not cell_features:
        return None, None
    centroids = np.array([td._polygon_centroid(f["geometry"]) for f in cell_features])
    raw_labels = ef.assign_labels(centroids, ann_path, verbose=False)
    own_classification = []
    for f in cell_features:
        cls = f["properties"].get("classification")
        own_classification.append(cls.get("name") if isinstance(cls, dict) else None)
    n = len(cell_features)

    names = sorted({k for f in cell_features for k in f["properties"].get("measurements", {})})
    m_arrays = {nm: np.full(n, np.nan, dtype=np.float64) for nm in names}
    for i, f in enumerate(cell_features):
        for nm, val in f["properties"].get("measurements", {}).items():
            if val is not None:
                m_arrays[nm][i] = val

    data = {
        "wsi_name": slide_name, "subtype": subtype, "nucleus_id": np.arange(n),
        "cx_wsi": centroids[:, 0].astype(np.float64), "cy_wsi": centroids[:, 1].astype(np.float64),
        "label": raw_labels, "own_classification": own_classification,
    }
    data.update(m_arrays)
    df = pd.DataFrame(data)
    return df, cell_features


def predict(model, X_s, device, batch_size=100000):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_s), batch_size):
            xb = torch.from_numpy(X_s[i:i + batch_size]).to(device)
            preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def fine_tune(pooled_state, feat_cols, hidden, dropout, X_tr, y_tr, X_val, y_val,
              device, epochs, lr, weight_decay, batch_size, seed, log_every=25):
    """Fresh copy of the pooled weights, fine-tuned on (X_tr, y_tr) only.
    Returns the trained model (best-val-loss epoch reloaded if a val set
    was given). Prints train/val loss+acc every log_every epochs so
    convergence (or lack of it) is actually visible."""
    torch.manual_seed(seed)
    model = td.MLP(len(feat_cols), 2, hidden, dropout).to(device)
    model.load_state_dict(pooled_state, strict=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()

    X_tr_t = torch.from_numpy(X_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr).to(device)
    has_val = X_val is not None and len(X_val) > 0
    if has_val:
        X_val_t = torch.from_numpy(X_val).to(device)
        y_val_t = torch.from_numpy(y_val).to(device)

    n = len(y_tr)
    bs = min(batch_size, n)
    steps_per_epoch = max(1, n // bs)

    best_val_loss = float("inf")
    best_state = None
    model.train()
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n, device=device)
        for i in range(steps_per_epoch):
            idx = perm[i * bs:(i + 1) * bs]
            xb, yb = X_tr_t[idx], y_tr_t[idx]
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            # Gradient clipping: if a batch still produces an unusually
            # large gradient (e.g. from a rare outlier example), cap its
            # effect on the weights rather than letting a single bad step
            # destabilize BatchNorm and cascade into the loss-explosion /
            # constant-output collapse pattern seen without this.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

        do_print = ep % log_every == 0 or ep == 1 or ep == epochs
        if do_print or has_val:
            model.eval()
            with torch.no_grad():
                train_logits = model(X_tr_t)
                train_loss = crit(train_logits, y_tr_t).item()
                train_acc = (train_logits.argmax(1) == y_tr_t).float().mean().item()
            if not np.isfinite(train_loss) or train_loss > 50.0:
                print(f"    WARNING: train_loss={train_loss:.4f} at epoch {ep} - training "
                      f"appears to be diverging (an outlier input value or too-high lr is "
                      f"the likely cause). Results from this slide should not be trusted.")
            msg = f"    epoch {ep:04d}  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}"
            if has_val:
                with torch.no_grad():
                    val_logits = model(X_val_t)
                    val_loss = crit(val_logits, y_val_t).item()
                    val_acc = (val_logits.argmax(1) == y_val_t).float().mean().item()
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(model.state_dict())
                msg += f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}" + ("  (best)" if is_best else "")
            if do_print:
                print(msg)
            model.train()

    if has_val and best_state is not None:
        model.load_state_dict(best_state)
        print(f"    reloaded best-val-loss epoch's weights for final save")
    model.eval()
    return model


def export_slide_geojson(df_infer, feats_all, pooled_pred_idx, overfit_pred_idx,
                          classes, out_root, wsi, subtype, excluded_df=None):
    i2c = {i: c for i, c in enumerate(classes)}
    pooled_pred = [i2c[i] for i in pooled_pred_idx]
    overfit_pred = [i2c[i] for i in overfit_pred_idx]

    mm_feats, pooled_feats, overfit_feats = [], [], []
    for row_pos, nid in enumerate(df_infer["nucleus_id"].to_numpy()):
        feat = feats_all[int(nid)]
        pp, op = pooled_pred[row_pos], overfit_pred[row_pos]

        mm_extra = {"overfit_pred": op, "pooled_model_pred": pp}
        cls = eg.classify_row(op, pp, classes)
        if cls is not None:
            mm_extra["classification"] = {"name": cls, "color": eg.CLASS_COLORS.get(cls, [128, 128, 128])}
        mm_feats.append(eg._new_feature(feat, mm_extra))

        pooled_feats.append(eg._new_feature(feat, {
            "classification": {"name": pp, "color": eg.RAW_CLASS_COLORS.get(pp, [128, 128, 128])}}))
        overfit_feats.append(eg._new_feature(feat, {
            "classification": {"name": op, "color": eg.RAW_CLASS_COLORS.get(op, [128, 128, 128])}}))

    # Nuclei excluded from inference (e.g. Ghost) keep their ORIGINAL
    # classification in every layer - never overwritten with a forced
    # model prediction, so they visibly remain "unclassified" w.r.t. the
    # Tumor/Non Tumor task instead of silently becoming one of the two.
    if excluded_df is not None and len(excluded_df):
        for _, row in excluded_df.iterrows():
            feat = feats_all[int(row["nucleus_id"])]
            orig_name = row["own_classification"] or "Unclassified"
            excl_extra = {"excluded_from_inference": True,
                          "classification": {"name": orig_name, "color": [160, 160, 160]}}
            mm_feats.append(eg._new_feature(feat, excl_extra))
            pooled_feats.append(eg._new_feature(feat, dict(excl_extra)))
            overfit_feats.append(eg._new_feature(feat, dict(excl_extra)))

    out_dir = os.path.join(out_root, subtype)
    os.makedirs(out_dir, exist_ok=True)
    for key, feats_out in (("pooled_vs_overfit_unannotated", mm_feats),
                            ("pooled_unannotated", pooled_feats),
                            ("overfit_unannotated", overfit_feats)):
        out_path = os.path.join(out_dir, f"{wsi}_{key}.geojson")
        tmp_path = out_path + ".tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump({"type": "FeatureCollection", "features": feats_out}, f)
            os.replace(tmp_path, out_path)  # atomic on the same filesystem -
                                             # out_path only ever exists fully written
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
        print(f"    saved [{key}] -> {out_path} ({len(feats_out)} features)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pooled_model", required=True)
    ap.add_argument("--feat_dir", default=td.REFERENCE_TEST_DIR,
                    help="Directory of whole-slide detection geojson(.gz) files. "
                         "Defaults to the reference/demo slide set.")
    ap.add_argument("--ann_dir", default=td.REFERENCE_ANN_DIR,
                    help="Matching annotation directory (point-in-polygon join). "
                         "Defaults to the reference set's annotations.")
    ap.add_argument("--feat_pattern", default=".geojson.gz")
    ap.add_argument("--subtypes", nargs="+", default=None)
    ap.add_argument("--slides", nargs="+", default=None,
                    help="Restrict to these slide names (default: every slide found).")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.5)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--val_frac", type=float, default=0.15,
                    help="Fraction of THIS SLIDE's annotated nuclei held out to monitor "
                         "fine-tuning fit. Set to 0 to fine-tune on all annotated nuclei.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--export_geojson_dir", required=True)
    ap.add_argument("--ckpt_dir", default=None,
                    help="If given, also save each slide's fine-tuned checkpoint here "
                         "as <ckpt_dir>/<slide>_overfit.pt.")
    ap.add_argument("--exclude_annotation_names", nargs="+", default=["Main"],
                    help="Annotation region names to exclude from the label join "
                         "(case-insensitive) - e.g. a big umbrella region that "
                         "contains everything else. Nuclei only covered by an "
                         "excluded region are treated as unannotated, not as "
                         "belonging to that region. Pass an empty list "
                         "(--exclude_annotation_names) to disable.")
    ap.add_argument("--exclude_from_inference", nargs="+", default=["Ghost"],
                    help="Nuclei whose OWN pre-existing classification (already "
                         "stored in the feature file, independent of any "
                         "annotation) matches one of these names (case-insensitive) "
                         "are NEVER fed to the model - they are excluded from "
                         "inference entirely and kept in the export with their "
                         "ORIGINAL classification, rather than being forced into "
                         "Tumor/Non Tumor. Pass an empty list to disable.")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt = torch.load(args.pooled_model, map_location=device, weights_only=False)
    feat_cols = ckpt["feature_cols"]
    mu, sd, medians = ckpt["mu"], ckpt["sd"], ckpt["medians"]
    classes = ckpt["classes"]
    hidden = tuple(ckpt.get("hidden", [48, 24, 12]))
    pooled_state = ckpt["model_state"]
    subtypes = args.subtypes or ckpt.get("subtypes") or td.DEFAULT_SUBTYPES
    print(f"Pooled checkpoint: {len(feat_cols)} features, classes={classes}, hidden={hidden}")

    pooled_model = td.MLP(len(feat_cols), len(classes), hidden, ckpt.get("dropout", 0.0)).to(device)
    pooled_model.load_state_dict(pooled_state, strict=True)
    pooled_model.eval()

    c2i = {c: i for i, c in enumerate(classes)}

    # Discover slides.
    slide_files = []
    for dirpath, _, filenames in os.walk(args.feat_dir):
        for fn in filenames:
            if not fn.endswith(args.feat_pattern):
                continue
            slide_name = fn[: -len(args.feat_pattern)]
            if args.slides and slide_name not in args.slides:
                continue
            st = td._infer_subtype_from_name(slide_name, subtypes)
            if st is None:
                continue
            slide_files.append((slide_name, st, os.path.join(dirpath, fn)))

    if not slide_files:
        raise SystemExit("No matching slide files found under --feat_dir.")
    print(f"Found {len(slide_files)} slide(s) to process: {[s[0] for s in slide_files]}")

    for slide_name, subtype, feat_path in slide_files:
        print(f"\n{'='*70}\nSlide: {slide_name}\n{'='*70}")
        # td._find_annotation_file now returns (path, matched_dir) - we
        # only need the path here.
        ann_path, _matched_dir = td._find_annotation_file(args.ann_dir, slide_name)
        if ann_path is None:
            print(f"  WARNING: no annotation file found for {slide_name} - skipping")
            continue

        join_ann_path, tmp_path = make_filtered_annotation_file(
            ann_path, args.exclude_annotation_names)
        try:
            df, feats_all = build_slide_dataframe(feat_path, join_ann_path, slide_name, subtype)
        finally:
            if tmp_path is not None:
                os.remove(tmp_path)
        if df is None:
            print(f"  WARNING: no features loaded for {slide_name} - skipping")
            continue
        # NOTE: train_deploy.py's feature-naming normalization
        # (_apply_feature_naming/_normalize_feature_name) has been REMOVED
        # entirely now that training and the reference set both use raw
        # ROI-style measurement names consistently - so it is no longer
        # called here either. Feature columns flow through exactly as
        # named in the geojson, and own_classification needs no special
        # protection since nothing renames columns anymore.
        df["_label"] = df["label"].apply(cf.collapse_label)

        annotated_mask = df["label"].notna()
        train_mask = annotated_mask & df["_label"].isin(classes)
        infer_mask = ~annotated_mask

        n_train, n_infer = int(train_mask.sum()), int(infer_mask.sum())
        print(f"  {len(df)} total nuclei | {n_train} annotated+usable (train) | "
              f"{n_infer} unannotated (inference target)")
        if n_train < 20:
            print(f"  WARNING: only {n_train} annotated nuclei - too few to fine-tune, skipping")
            continue
        if n_infer == 0:
            print(f"  WARNING: 0 unannotated nuclei - nothing to export, skipping")
            continue

        # Standardize using the CHECKPOINT's own mu/sd/medians (not
        # recomputed per slide) so both models see inputs in the same
        # space - isolates the comparison to WEIGHTS, not preprocessing.
        def to_X(sub_df):
            X = sub_df.reindex(columns=feat_cols).astype(np.float64)
            X = X.fillna(pd.Series(medians, index=feat_cols))
            X_s = (X.to_numpy() - mu) / sd
            # Clamp extreme standardized values - a single measurement far
            # outside the range the checkpoint's mu/sd were computed from
            # (e.g. an outlier Haralick/texture value on a badly segmented
            # or artifact nucleus) can otherwise produce a huge input that
            # destabilizes BatchNorm and blows up training loss into the
            # hundreds/thousands within a few epochs, collapsing the model
            # to a constant single-class output. +/-20 std devs is far
            # beyond any real signal but stops a literal numerical outlier
            # from wrecking the whole run.
            X_s = np.clip(X_s, -20.0, 20.0)
            return X_s.astype(np.float32)

        train_df = df[train_mask].reset_index(drop=True)
        infer_df_full = df[infer_mask].reset_index(drop=True)

        # Nuclei whose OWN pre-existing classification matches an excluded
        # name (e.g. "Ghost") are pulled out here, BEFORE prediction, so
        # the model never sees them and never has to force a Tumor/Non
        # Tumor guess onto them. They still get exported (see
        # export_slide_geojson below) but with their ORIGINAL
        # classification kept, not a model prediction.
        exclude_lower = {n.strip().lower() for n in args.exclude_from_inference}
        own_cls = infer_df_full["own_classification"].fillna("").str.strip().str.lower()
        excluded_mask = own_cls.isin(exclude_lower)
        infer_df = infer_df_full[~excluded_mask].reset_index(drop=True)
        excluded_df = infer_df_full[excluded_mask].reset_index(drop=True)
        if len(excluded_df):
            print(f"  Excluding {len(excluded_df)} nuclei from inference "
                  f"(own classification in {sorted(exclude_lower)}) - kept in "
                  f"export with their original classification unchanged.")
        if len(infer_df) == 0:
            print(f"  WARNING: 0 nuclei left to classify after exclusions - skipping export")
            continue

        X_all_train = to_X(train_df)
        y_all_train = train_df["_label"].map(c2i).to_numpy()

        X_val = y_val = None
        if args.val_frac > 0 and n_train >= 20:
            from sklearn.model_selection import train_test_split
            counts = pd.Series(y_all_train).value_counts()
            can_stratify = (counts >= 2).all() and len(counts) >= 2
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_all_train, y_all_train, test_size=args.val_frac,
                random_state=args.seed, stratify=y_all_train if can_stratify else None)
        else:
            X_tr, y_tr = X_all_train, y_all_train

        overfit_model = fine_tune(pooled_state, feat_cols, hidden, args.dropout,
                                   X_tr, y_tr, X_val, y_val, device,
                                   args.epochs, args.lr, args.weight_decay,
                                   args.batch_size, args.seed)

        # Did it actually overfit? Compare BOTH models' accuracy on the
        # annotated training data itself. The overfit model should score
        # far higher than the pooled model does on the exact data it was
        # just fine-tuned on - that gap IS the overfitting, made concrete.
        pooled_pred_on_train = predict(pooled_model, X_all_train, device)
        overfit_pred_on_train = predict(overfit_model, X_all_train, device)
        pooled_train_acc = float((pooled_pred_on_train == y_all_train).mean())
        overfit_train_acc = float((overfit_pred_on_train == y_all_train).mean())
        print(f"  Accuracy on THIS SLIDE's annotated (training) nuclei:")
        print(f"    pooled model (never saw this fine-tuning):  {pooled_train_acc:.4f}")
        print(f"    overfit model (fine-tuned on exactly this): {overfit_train_acc:.4f}")
        if overfit_train_acc - pooled_train_acc < 0.02:
            print(f"    NOTE: overfit model barely improved over pooled on its own "
                  f"training data ({overfit_train_acc:.4f} vs {pooled_train_acc:.4f}) - "
                  f"fine-tuning may not have converged. Check --lr/--epochs.")

        if args.ckpt_dir:
            os.makedirs(args.ckpt_dir, exist_ok=True)
            out_ckpt_path = os.path.join(args.ckpt_dir, f"{slide_name}_overfit.pt")
            torch.save({
                "model_state": overfit_model.state_dict(), "classes": classes,
                "feature_cols": feat_cols, "mu": mu, "sd": sd, "medians": medians,
                "hidden": list(hidden), "dropout": args.dropout, "subtypes": subtypes,
                "base_checkpoint": os.path.abspath(args.pooled_model),
                "note": f"DEMO ARTIFACT - overfit on {slide_name}'s annotated nuclei only.",
            }, out_ckpt_path)
            print(f"  saved fine-tuned checkpoint -> {out_ckpt_path}")

        X_infer = to_X(infer_df)
        pooled_pred_idx = predict(pooled_model, X_infer, device)
        overfit_pred_idx = predict(overfit_model, X_infer, device)
        agree = float((pooled_pred_idx == overfit_pred_idx).mean())
        print(f"  Pooled vs overfit agreement on UNANNOTATED nuclei (excluding "
              f"{sorted(exclude_lower)}): {agree:.4f} "
              f"({int((pooled_pred_idx == overfit_pred_idx).sum())}/{len(infer_df)})")

        export_slide_geojson(infer_df, feats_all, pooled_pred_idx, overfit_pred_idx,
                              classes, args.export_geojson_dir, slide_name, subtype,
                              excluded_df=excluded_df)

    print(f"\nDone. Exports under {args.export_geojson_dir}/<subtype>/ - "
          f"original files under {args.feat_dir}/{args.ann_dir} were not modified.")


if __name__ == "__main__":
    main()