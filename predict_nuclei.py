"""
Called by QuPath (predict_in_qupath.groovy). Reads a CSV of exported
detection measurements, runs the pooled model, writes predictions to JSON.

Usage:
    python predict_nuclei.py --model pooled_model.pt --csv in.csv --out out.json
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


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
    ap.add_argument("--model", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_confidence", type=float, default=0.0,
                    help="Below this softmax probability, leave the detection "
                         "unclassified instead of guessing.")
    args = ap.parse_args()

    try:
        ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"ERROR: could not load model {args.model}: {e}", file=sys.stderr)
        sys.exit(1)

    classes = ckpt["classes"]
    qupath_names = ckpt.get("qupath_names", classes)
    feat_cols = ckpt["feature_cols"]
    mu, sd = np.asarray(ckpt["mu"]), np.asarray(ckpt["sd"])
    medians = np.asarray(ckpt["medians"])

    df = pd.read_csv(args.csv)
    print(f"Read {len(df)} detections from {args.csv}")

    # The model needs exactly the features it was trained on, in order.
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        print(f"ERROR: CSV is missing {len(missing)} required features, e.g. "
              f"{missing[:5]}", file=sys.stderr)
        print("Check that intensity + smoothed features were computed on this "
              "image.", file=sys.stderr)
        sys.exit(1)

    X = df[feat_cols].to_numpy(dtype=np.float64)
    nan_frac = np.isnan(X).mean()
    if nan_frac > 0.5:
        print(f"ERROR: {100*nan_frac:.0f}% of feature values are missing. "
              f"Features were probably never computed on this image.", file=sys.stderr)
        sys.exit(1)
    if nan_frac > 0:
        print(f"Note: filling {100*nan_frac:.2f}% missing values with training medians")
    X = np.where(np.isnan(X), medians, X)

    sd_safe = sd.copy()
    sd_safe[sd_safe == 0] = 1.0
    X = ((X - mu) / sd_safe).astype(np.float32)

    model = MLP(len(feat_cols), len(classes),
                tuple(ckpt.get("hidden", [48, 24, 12])), ckpt.get("dropout", 0.3))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    probs = []
    with torch.no_grad():
        for i in range(0, len(X), 8192):
            out = model(torch.from_numpy(X[i:i + 8192]))
            probs.append(torch.softmax(out, dim=1).numpy())
    probs = np.concatenate(probs) if probs else np.zeros((0, len(classes)))

    pred_idx = probs.argmax(1)
    confidence = probs.max(1)

    records = []
    n_low = 0
    for i in range(len(df)):
        conf = float(confidence[i])
        if conf < args.min_confidence:
            name = None
            n_low += 1
        else:
            name = qupath_names[pred_idx[i]]
        records.append({
            "cx": float(df.iloc[i]["cx_wsi"]),
            "cy": float(df.iloc[i]["cy_wsi"]),
            "cls": name,
            "conf": round(conf, 4),
        })

    payload = {
        "model": args.model,
        "classes": classes,
        "qupath_names": qupath_names,
        "n": len(records),
        "n_below_threshold": n_low,
        "min_confidence": args.min_confidence,
        "predictions": records,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f)

    counts = {}
    for r in records:
        counts[r["cls"] or "(unclassified)"] = counts.get(r["cls"] or "(unclassified)", 0) + 1
    print(f"Wrote {len(records)} predictions -> {args.out}")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v}")
    print(f"Mean confidence: {confidence.mean():.3f}")


if __name__ == "__main__":
    main()