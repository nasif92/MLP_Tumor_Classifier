"""
Trains ONE pooled model on all five subtypes and saves it for deployment.

Unlike cross_subtype.py (which trains a throwaway model per CV fold), this
fits a single model on everything and saves the weights plus the exact
preprocessing needed at inference time.

Usage:
    python train_deploy_model.py --root . --out pooled_model.pt
    python train_deploy_model.py --root . --holdout_frac 0.1   # sanity check
"""
import argparse
import glob
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

NON_FEATURE_COLS = {
    "wsi_name", "nucleus_id", "cx_wsi", "cy_wsi", "label", "is_ground_truth",
    "containing_annotation_ids", "containing_annotation_classes",
    "annotation_nesting_depth",
}
DEFAULT_SUBTYPES = ["TNBC", "Cervix", "HNSCC", "NSCLC", "UpperGI"]

# QuPath class names to write back, keyed by the collapsed training label.
QUPATH_NAMES = {
    "Tumor": "Tumor",
    "Stroma": "Stroma",
    "Immune": "Immune cells",
    "Other": "Other",
    "Normal": "Normal-Glands",
}


def collapse_label(raw):
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    raw = str(raw).strip()
    if raw == "" or raw.lower().startswith("ignore"):
        return None
    if raw == "Tumor":
        return "Tumor"
    if raw == "Stroma":
        return "Stroma"
    if raw == "Immune cells":
        return "Immune"
    if raw.startswith("Normal"):
        return "Normal"
    if raw in ("Necrosis", "Other"):
        return "Other"
    return None


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
    ap.add_argument("--root", default=".")
    ap.add_argument("--subtypes", nargs="+", default=DEFAULT_SUBTYPES)
    ap.add_argument("--classes", nargs="+",
                    default=["Tumor", "Stroma", "Immune", "Other"])
    ap.add_argument("--out", default="pooled_model.pt")
    ap.add_argument("--holdout_frac", type=float, default=0.0,
                    help="Hold out this fraction of SLIDES for a sanity check. "
                         "Use 0 for the final deployment model (train on everything).")
    ap.add_argument("--hidden", nargs="+", type=int, default=[48, 24, 12])
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--reweight_power", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frames = []
    for st in args.subtypes:
        paths = sorted(glob.glob(os.path.join(args.root, st, "*_gt_measurements.csv")))
        if not paths:
            print(f"  {st}: no CSVs, skipping")
            continue
        sub = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
        sub["subtype"] = st
        sub["slide_uid"] = st + ":" + sub["wsi_name"].astype(str)
        frames.append(sub)
        print(f"  {st}: {len(paths)} slides, {len(sub)} nuclei")
    if not frames:
        raise SystemExit("No data found.")

    df = pd.concat(frames, ignore_index=True)
    df["_label"] = df["label"].apply(collapse_label)
    df = df.dropna(subset=["_label"])
    df = df[df["_label"].isin(args.classes)].reset_index(drop=True)

    feat_cols = [c for c in df.columns
                 if c not in NON_FEATURE_COLS and not c.startswith("_")
                 and c not in ("subtype", "slide_uid")
                 and pd.api.types.is_numeric_dtype(df[c])
                 and df[c].notna().any()]
    print(f"\n{len(df)} nuclei | {df['slide_uid'].nunique()} slides | {len(feat_cols)} features")
    print("\nClass counts:\n", df["_label"].value_counts())

    # Median imputation - stored in the checkpoint so inference fills gaps
    # identically rather than using the new slide's own medians.
    medians = df[feat_cols].median()
    df[feat_cols] = df[feat_cols].fillna(medians)

    classes = list(args.classes)
    c2i = {c: i for i, c in enumerate(classes)}
    X_all = df[feat_cols].to_numpy(dtype=np.float64)
    y_all = df["_label"].map(c2i).to_numpy()

    if args.holdout_frac > 0:
        slides = sorted(df["slide_uid"].unique())
        rng = np.random.RandomState(args.seed)
        rng.shuffle(slides)
        n_hold = max(1, int(len(slides) * args.holdout_frac))
        hold = set(slides[:n_hold])
        te = df["slide_uid"].isin(hold).to_numpy()
        print(f"\nHolding out {n_hold} slides for sanity check: {sorted(hold)}")
    else:
        te = np.zeros(len(df), dtype=bool)
        print("\nTraining on ALL slides (no holdout) - deployment model.")

    tr = ~te
    X_tr, y_tr = X_all[tr], y_all[tr]

    mu, sd = X_tr.mean(0), X_tr.std(0)
    sd[sd == 0] = 1.0
    X_tr_s = ((X_tr - mu) / sd).astype(np.float32)

    ds = TensorDataset(torch.from_numpy(X_tr_s), torch.from_numpy(y_tr))
    counts = Counter(y_tr.tolist())
    w = np.array([1.0 / (counts[int(t)] ** args.reweight_power) for t in y_tr])
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                    num_samples=len(w), replacement=True)
    loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler, drop_last=True)

    model = MLP(len(feat_cols), len(classes), tuple(args.hidden), args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss()

    for ep in range(1, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * xb.size(0)
            n += xb.size(0)
        if ep % 10 == 0 or ep == 1:
            print(f"  epoch {ep:03d}  loss {tot/n:.4f}")

    if te.any():
        from sklearn.metrics import classification_report, f1_score
        X_te_s = ((X_all[te] - mu) / sd).astype(np.float32)
        model.eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(X_te_s).to(device)).argmax(1).cpu().numpy()
        print("\n=== HOLDOUT SANITY CHECK ===")
        print(f"acc={np.mean(pred == y_all[te]):.4f}  "
              f"macro_f1={f1_score(y_all[te], pred, average='macro', zero_division=0):.4f}")
        print(classification_report(y_all[te], pred, labels=list(range(len(classes))),
                                    target_names=classes, zero_division=0))

    torch.save({
        "model_state": model.state_dict(),
        "classes": classes,
        "qupath_names": [QUPATH_NAMES.get(c, c) for c in classes],
        "feature_cols": feat_cols,
        "mu": mu, "sd": sd,
        "medians": medians[feat_cols].to_numpy(),
        "hidden": args.hidden, "dropout": args.dropout,
        "subtypes": args.subtypes,
        "n_train_nuclei": int(tr.sum()),
        "n_train_slides": int(df.loc[tr, "slide_uid"].nunique()),
    }, args.out)
    print(f"\nSaved -> {args.out}")
    print(f"  {len(feat_cols)} features, classes: {classes}")
    print(f"  QuPath names: {[QUPATH_NAMES.get(c, c) for c in classes]}")


if __name__ == "__main__":
    main()