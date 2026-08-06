"""
Trains an MLP on QuPath nucleus measurements, with two fixes for the batch
effect that made the previous version fail (train acc 1.00 / val acc 0.24):

1. PER-SLIDE NORMALIZATION - z-score each feature within each slide. The
   diagnostic showed 30/50 features vary more between slides than between
   classes (staining differences), so raw values encode slide identity.
   Normalizing within a slide removes that offset and keeps only the
   relative differences between cells on the SAME slide.

2. LEAVE-ONE-SLIDE-OUT CV - with 11 slides of very unequal size (D1 alone
   is 44% of the data), a single held-out split is dominated by whichever
   slide you pick. LOSO reports mean +/- std across all slides.

Usage:
    python train_mlp_v2.py --root /path/to/project --classes Tumor Stroma Immune
    python train_mlp_v2.py --root /path/to/project --no_slide_norm   # ablation
    python train_mlp_v2.py --root /path/to/project --model rf        # RF instead
"""
import argparse
import glob
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


def load_data(root, keep_classes):
    paths = sorted(glob.glob(os.path.join(root,
                                          "*_gt_measurements.csv")))
    if not paths:
        raise SystemExit(f"No CSVs under {root}/ground_truth_measurements/")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df["_label"] = df["label"].apply(collapse_label)
    df = df.dropna(subset=["_label"])
    before = len(df)
    df = df[df["_label"].isin(keep_classes)].reset_index(drop=True)
    print(f"Loaded {before} labeled nuclei; {len(df)} after keeping {keep_classes}")
    return df


def get_feature_cols(df):
    return [c for c in df.columns
            if c not in NON_FEATURE_COLS and not c.startswith("_")
            and pd.api.types.is_numeric_dtype(df[c])
            and df[c].notna().any()
            and df[c].std(skipna=True) not in (0, np.nan)]


def normalize_per_slide(df, feat_cols):
    """Z-score each feature within each slide, so absolute staining level
    (which differs slide to slide) is removed and only relative differences
    between cells on the same slide remain."""
    out = df.copy()
    for slide, idx in df.groupby("wsi_name").groups.items():
        block = out.loc[idx, feat_cols]
        mu = block.mean()
        sd = block.std().replace(0, 1.0).fillna(1.0)
        out.loc[idx, feat_cols] = ((block - mu) / sd).values
    return out


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


def train_one_fold(X_tr, y_tr, X_va, y_va, n_classes, args, device):
    if args.model == "rf":
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                    min_samples_leaf=5, random_state=args.seed, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        return rf.predict(X_va), rf.predict(X_tr)

    tr_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    counts = Counter(y_tr.tolist())
    w = np.array([1.0 / (counts[int(t)] ** args.reweight_power) for t in y_tr])
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                    num_samples=len(w), replacement=True)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, sampler=sampler,
                           drop_last=len(tr_ds) > args.batch_size)

    model = MLP(X_tr.shape[1], n_classes, tuple(args.hidden), args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss()

    for _ in range(args.epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        va_pred = model(torch.from_numpy(X_va).to(device)).argmax(1).cpu().numpy()
        tr_pred = model(torch.from_numpy(X_tr).to(device)).argmax(1).cpu().numpy()
    return va_pred, tr_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--classes", nargs="+", default=["Tumor", "Stroma", "Immune"],
                    help="Classes to keep. Normal/Other appear on only 2 slides each "
                         "and cannot be learned under a slide split.")
    ap.add_argument("--model", choices=["mlp", "rf"], default="mlp")
    ap.add_argument("--no_slide_norm", action="store_true",
                    help="Disable per-slide normalization (ablation)")
    ap.add_argument("--min_class_slides", type=int, default=3,
                    help="A fold's val slide must have >=1 of a class for it to count")
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

    classes = list(args.classes)
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    df = load_data(args.root, classes)

    feat_cols = get_feature_cols(df)
    print(f"Features: {len(feat_cols)}")

    # Impute NaNs with global medians before normalizing.
    df[feat_cols] = df[feat_cols].fillna(df[feat_cols].median())

    if not args.no_slide_norm:
        df = normalize_per_slide(df, feat_cols)
        print("Per-slide normalization: ON")
    else:
        print("Per-slide normalization: OFF (ablation)")

    print("\nClass counts per slide:")
    pivot = df.pivot_table(index="wsi_name", columns="_label", aggfunc="size", fill_value=0)
    print(pivot.to_string())

    X_all = df[feat_cols].to_numpy(dtype=np.float32)
    y_all = df["_label"].map(cls_to_idx).to_numpy()
    slides = df["wsi_name"].to_numpy()
    unique_slides = sorted(set(slides))

    from sklearn.metrics import f1_score, confusion_matrix

    print(f"\n=== LEAVE-ONE-SLIDE-OUT CV ({args.model.upper()}) ===")
    print(f"{'val slide':>10} {'n':>7} {'classes':>9} {'train_acc':>10} "
          f"{'val_acc':>9} {'val_f1':>9}")

    results, all_pred, all_true = [], [], []
    for vs in unique_slides:
        va_mask = slides == vs
        tr_mask = ~va_mask
        n_va_classes = len(set(y_all[va_mask]))
        if n_va_classes < 2:
            print(f"{vs:>10} {va_mask.sum():>7} {n_va_classes:>9}   skipped "
                  f"(needs >=2 classes to score)")
            continue

        X_tr = X_all[tr_mask]
        X_va = X_all[va_mask]
        if args.no_slide_norm:
            # Without per-slide norm, still standardize on train statistics.
            mu, sd = X_tr.mean(0), X_tr.std(0)
            sd[sd == 0] = 1.0
            X_tr, X_va = (X_tr - mu) / sd, (X_va - mu) / sd

        va_pred, tr_pred = train_one_fold(X_tr, y_all[tr_mask], X_va, y_all[va_mask],
                                          len(classes), args, device)
        tr_acc = (tr_pred == y_all[tr_mask]).mean()
        va_acc = (va_pred == y_all[va_mask]).mean()
        va_f1 = f1_score(y_all[va_mask], va_pred, average="macro",
                         labels=list(range(len(classes))), zero_division=0)
        print(f"{vs:>10} {va_mask.sum():>7} {n_va_classes:>9} {tr_acc:>10.4f} "
              f"{va_acc:>9.4f} {va_f1:>9.4f}")
        results.append((vs, va_acc, va_f1))
        all_pred.extend(va_pred)
        all_true.extend(y_all[va_mask])

    if not results:
        raise SystemExit("No scoreable folds.")

    accs = np.array([r[1] for r in results])
    f1s = np.array([r[2] for r in results])
    print(f"\nMean across {len(results)} folds: "
          f"acc={accs.mean():.4f}+/-{accs.std():.4f}  "
          f"macro_f1={f1s.mean():.4f}+/-{f1s.std():.4f}")

    # Pooled confusion matrix - every nucleus predicted exactly once, by a
    # model that never saw its slide.
    all_pred, all_true = np.array(all_pred), np.array(all_true)
    print(f"\n=== POOLED (n={len(all_true)}) ===")
    print(f"acc={np.mean(all_pred == all_true):.4f}  "
          f"macro_f1={f1_score(all_true, all_pred, average='macro', zero_division=0):.4f}")
    from sklearn.metrics import classification_report
    print(classification_report(all_true, all_pred, labels=list(range(len(classes))),
                                target_names=classes, zero_division=0))
    cm = confusion_matrix(all_true, all_pred, labels=list(range(len(classes))))
    print("Confusion (rows=true, cols=pred): " + "  ".join(classes))
    for i, r in enumerate(cm):
        print(f"{classes[i]:>8}: " + " ".join(f"{v:7d}" for v in r))


if __name__ == "__main__":
    main()