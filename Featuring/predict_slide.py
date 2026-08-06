"""
Classify nuclei end-to-end, without QuPath.

    WSI + detection GeoJSON -> features -> MLP -> classified GeoJSON

The output GeoJSON carries QuPath classifications and a per-nucleus
prediction confidence, so it can be dragged straight into QuPath
(File > Import > Objects) and viewed.

Usage:
    python predict_slide.py --wsi slide.svs --geojson dets.geojson.gz \\
        --model pooled_model.pt --out slide_classified.geojson

    # only classify nuclei inside annotated regions
    python predict_slide.py ... --annotations regions.geojson

    # leave low-confidence nuclei unclassified
    python predict_slide.py ... --min_confidence 0.6
"""
import argparse
import gzip
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import extract_features as ef

# QuPath display colours, by training class name.
CLASS_COLORS = {
    "Tumor": [255, 0, 0],
    "Stroma": [0, 200, 0],
    "Immune": [0, 0, 255],
    "Immune cells": [0, 0, 255],
    "Normal": [0, 255, 255],
    "Other": [150, 0, 150],
}
# Training label -> the class name QuPath should show.
QUPATH_NAMES = {"Immune": "Immune cells"}


class MLP(nn.Module):
    def __init__(self, n_features, n_classes, hidden=(48, 24, 12), dropout=0.3):
        super().__init__()
        layers, prev = [], n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi", required=True)
    ap.add_argument("--geojson", required=True, help="Detection polygons")
    ap.add_argument("--model", default="pooled_model.pt")
    ap.add_argument("--out", required=True, help="Output .geojson (or .geojson.gz)")
    ap.add_argument("--annotations", default=None,
                    help="Restrict to nuclei inside these regions")
    ap.add_argument("--min_confidence", type=float, default=0.0,
                    help="Below this, leave the nucleus unclassified")
    ap.add_argument("--mpp", type=float, default=None)
    ap.add_argument("--csv_out", default=None,
                    help="Also write the feature table + predictions here")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    feat_cols = ckpt["feature_cols"]
    mu = np.asarray(ckpt["mu"])
    sd = np.asarray(ckpt["sd"])
    sd[sd == 0] = 1.0
    medians = np.asarray(ckpt["medians"])
    print(f"Model: {args.model}")
    print(f"  classes  : {classes}")
    print(f"  features : {len(feat_cols)}")
    print(f"  trained on {ckpt.get('n_train_nuclei','?')} nuclei / "
          f"{ckpt.get('n_train_slides','?')} slides")

    name = os.path.splitext(os.path.basename(args.wsi))[0]
    print(f"\n=== {name} ===")

    t0 = time.time()
    polys = ef.polygons_from_geojson(args.geojson)
    print(f"  {len(polys)} detections ({time.time()-t0:.1f}s)")
    if not polys:
        raise SystemExit("No polygons found.")

    centroids_all = np.array([p.mean(axis=0) for p in polys])
    centroids_every_detection = centroids_all.copy()

    keep_idx = np.arange(len(polys))
    if args.annotations:
        labels = ef.assign_labels(centroids_all, args.annotations, verbose=False)
        keep_idx = np.array([i for i, l in enumerate(labels) if l is not None])
        if keep_idx.size == 0:
            raise SystemExit("No nuclei inside the given annotations.")
        print(f"  {keep_idx.size} inside annotations "
              f"(skipping {len(polys) - keep_idx.size})")
        polys = [polys[i] for i in keep_idx]
        centroids_all = centroids_all[keep_idx]

    print("  extracting features...")
    with ef.NucleusFeatureExtractor(args.wsi, mpp=args.mpp) as ext:
        base, centroids = ext.extract_many(polys, centroids=centroids_all,
                                           progress_every=20000)
        mpp = ext.mpp
    # Neighbour counts use every detection, matching how the model was trained.
    df = ef.add_smoothed_features(base, centroids, mpp,
                                  all_centroids_px=centroids_every_detection)

    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Model expects {len(missing)} features this extractor did not "
            f"produce, e.g. {missing[:5]}\n"
            "The model was probably trained on a different feature set.")

    X = df[feat_cols].to_numpy(dtype=np.float64)
    X = np.where(np.isnan(X), medians, X)
    X = ((X - mu) / sd).astype(np.float32)

    model = MLP(len(feat_cols), len(classes),
                tuple(ckpt.get("hidden", [48, 24, 12])),
                ckpt.get("dropout", 0.3)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    probs = []
    with torch.no_grad():
        for i in range(0, len(X), 8192):
            out = model(torch.from_numpy(X[i:i + 8192]).to(device))
            probs.append(torch.softmax(out, dim=1).cpu().numpy())
    probs = np.concatenate(probs)
    pred = probs.argmax(1)
    conf = probs.max(1)

    counts = {}
    features = []
    for k, poly in enumerate(polys):
        c = float(conf[k])
        if c < args.min_confidence:
            cls_name = None
        else:
            cls_name = classes[pred[k]]
            cls_name = QUPATH_NAMES.get(cls_name, cls_name)
        counts[cls_name or "(unclassified)"] = counts.get(cls_name or "(unclassified)", 0) + 1

        ring = np.asarray(poly, dtype=float)
        if not np.allclose(ring[0], ring[-1]):     # GeoJSON rings must close
            ring = np.vstack([ring, ring[:1]])

        props = {"objectType": "detection",
                 "measurements": {"pred_confidence": round(c, 4)}}
        if cls_name is not None:
            props["classification"] = {
                "name": cls_name,
                "color": CLASS_COLORS.get(cls_name, [128, 128, 128]),
            }
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon",
                         "coordinates": [ring.round(2).tolist()]},
            "properties": props,
        })

    payload = {"type": "FeatureCollection", "features": features}
    opener = gzip.open if args.out.endswith(".gz") else open
    with opener(args.out, "wt", encoding="utf-8") as f:
        json.dump(payload, f)

    print(f"\n  wrote {len(features)} classified nuclei -> {args.out}")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<20} {v:>8}  ({100*v/len(features):.1f}%)")
    print(f"  mean confidence: {conf.mean():.3f}")
    lo = (conf < 0.6).mean()
    print(f"  below 0.6 confidence: {100*lo:.1f}%")

    if args.csv_out:
        df.insert(0, "cy_wsi", centroids[:, 1])
        df.insert(0, "cx_wsi", centroids[:, 0])
        df.insert(0, "wsi_name", name)
        df["predicted"] = [classes[p] for p in pred]
        df["confidence"] = conf
        df.to_csv(args.csv_out, index=False)
        print(f"  features + predictions -> {args.csv_out}")

    print("\nTo view: QuPath > File > Import > Objects, then select this file.")


if __name__ == "__main__":
    main()