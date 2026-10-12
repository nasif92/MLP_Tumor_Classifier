"""
Feature-group Mixture-of-Experts for nucleus Tumor / Non-Tumor classification.

Design
------
Each expert is a small MLP that sees ONLY its own block of handcrafted
features (shape, size, intensity, texture, optionally context and a learned
embedding). A gating network looks at all features plus the slide's stain
(ER / PR / Ki67) and produces a per-nucleus softmax over experts. The
gate-weighted sum of expert embeddings goes through a shared classifier head.

    x ──┬─ shape cols  ─► E_shape  ─► z1 ─┐
        ├─ size cols   ─► E_size   ─► z2 ─┤   fused = Σ g_e · z_e  ─► head ─► logits
        ├─ intensity   ─► E_int    ─► z3 ─┤
        ├─ texture     ─► E_tex    ─► z4 ─┘
        └─ all cols + stain emb ─► gate ─► g (softmax over experts)

Training extras (all optional, all off at inference):
  * expert dropout     - whole experts masked per nucleus, gate renormalized,
                         so the head can't lean on one expert.
  * auxiliary heads    - each expert has its own linear classifier trained
                         with a small weight, so every expert stays
                         independently discriminative (this is what keeps
                         the members of an ensemble "accurate" in Windeatt's
                         accuracy/diversity sense).
  * online bagging     - per-expert Poisson(1) sample weights on the aux
                         losses: an online approximation of training each
                         base MLP on its own bootstrap replicate (Breiman's
                         bagging, as used for MLP ensembles in Windeatt 2008),
                         to inject diversity between experts.
  * load balancing     - KL(mean gate usage || uniform) across the batch,
                         which discourages collapse onto one expert while
                         still allowing per-nucleus specialization.
  * stain dropout      - the stain index is randomly replaced by an
                         "unknown" token so the model still works when the
                         inference script doesn't pass a stain.

Diagnostics (report_experts):
  per-expert accuracy, pairwise Yule's Q between experts (Windeatt eq. 6 /
  Kuncheva & Whitaker 2003 - low or negative Q = diverse errors, which is
  where combining helps), and mean gate weight by stain x class.
"""
import re
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# Feature -> expert assignment. Ordered: the FIRST matching rule wins, so
# "Smoothed: 50 µm: Area µm^2" goes to context, not size. Edit freely; the
# resulting groups are saved in the checkpoint, so a model always rebuilds
# with the grouping it was trained with.
DERIVED_ASPECT = "Derived: Aspect ratio"

GROUP_RULES = [
    # Neighbourhood features, split by content (radius 50 and 75 together).
    ("context_morph", re.compile(r"^Smoothed: \d+ µm: (?!ROI:)")),
    ("context_stain", re.compile(r"^Smoothed: \d+ µm: ROI:")),
    ("embedding",     re.compile(r"^(emb|dino|uni|deep)[\s_:\-]*\d+$", re.IGNORECASE)),
    # One expert per stain channel: intensity + Haralick + entropy of that channel.
    ("dab",           re.compile(r"DAB")),
    ("hematoxylin",   re.compile(r"Hematoxylin")),
    ("od_sum",        re.compile(r"OD Sum")),
    # Shape and size merged - 2 and 4 features were too few for separate experts.
    ("morphology",    re.compile(r"^(Area µm\^2|Length µm|Max diameter µm|Min diameter µm|Perimeter.*|"
                                 r"Circularity|Solidity|Eccentricity|" + re.escape(DERIVED_ASPECT) + r")$")),
]


def add_derived_features(df):
    """Adds features that don't exist in the QuPath export but help the shape
    expert. Min/Max diameter is an aspect-ratio proxy (1 = round, -> 0 =
    elongated); eccentricity is a monotone function of it, so one is enough.
    MUST also be applied at inference before selecting feature_cols, or the
    column will be median-filled (constant) for every nucleus."""
    if DERIVED_ASPECT not in df.columns and {"Min diameter µm", "Max diameter µm"} <= set(df.columns):
        mx = df["Max diameter µm"].astype(float)
        df[DERIVED_ASPECT] = (df["Min diameter µm"].astype(float) / mx.where(mx > 0)).clip(0, 1)
    return df


def assign_groups(feat_cols, drop_groups=()):
    """-> OrderedDict name -> list of column indices into feat_cols. Anything
    matching no rule goes to 'other' (so nothing is silently discarded)."""
    groups = OrderedDict((name, []) for name, _ in GROUP_RULES)
    groups["other"] = []
    for i, c in enumerate(feat_cols):
        for name, rx in GROUP_RULES:
            if rx.search(c):
                groups[name].append(i)
                break
        else:
            groups["other"].append(i)
    return OrderedDict((k, v) for k, v in groups.items() if v and k not in set(drop_groups))


def print_groups(groups, feat_cols):
    print("\nMoE experts (feature-group -> #features):")
    for name, idx in groups.items():
        ex = ", ".join(feat_cols[i].replace("ROI: 0.50 µm per pixel: ", "") for i in idx[:4])
        print(f"  {name:<10} {len(idx):>3}  e.g. {ex}{' ...' if len(idx) > 4 else ''}")


# --------------------------------------------------------------------------
class Expert(nn.Module):
    """Small per-group encoder. LayerNorm (not BatchNorm) so behaviour doesn't
    depend on batch composition - batches here are reweighted samples, and
    inference may run slide by slide."""

    def __init__(self, n_in, d, hidden, dropout):
        super().__init__()
        h = max(hidden, 2 * n_in) if n_in > 16 else hidden
        self.net = nn.Sequential(
            nn.Linear(n_in, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h, d), nn.LayerNorm(d), nn.GELU())

    def forward(self, x):
        return self.net(x)


class MoEClassifier(nn.Module):
    def __init__(self, n_features, n_classes, groups, n_subtypes, d=32,
                 expert_hidden=32, gate_hidden=64, head_hidden=32, dropout=0.2,
                 subtype_dim=8, use_subtype=True, gate_temp=1.0,
                 expert_dropout=0.15, subtype_drop=0.2):
        super().__init__()
        self.group_names = list(groups.keys())
        self.n_experts = len(self.group_names)
        for k, idx in groups.items():
            self.register_buffer(f"idx_{k}", torch.as_tensor(idx, dtype=torch.long), persistent=True)
        self.experts = nn.ModuleList(
            Expert(len(groups[k]), d, expert_hidden, dropout) for k in self.group_names)
        self.aux_heads = nn.ModuleList(nn.Linear(d, n_classes) for _ in self.group_names)

        self.use_subtype = use_subtype
        self.n_subtypes = n_subtypes                  # index n_subtypes == "unknown"
        self.subtype_emb = nn.Embedding(n_subtypes + 1, subtype_dim) if use_subtype else None
        g_in = n_features + (subtype_dim if use_subtype else 0)
        self.gate = nn.Sequential(nn.Linear(g_in, gate_hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(gate_hidden, self.n_experts))
        self.head = nn.Sequential(nn.Linear(d, head_hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(head_hidden, n_classes))
        self.feat_dim = d
        self.gate_temp = gate_temp
        self.expert_dropout = expert_dropout
        self.subtype_drop = subtype_drop

    # ---------------------------------------------------------------- pieces
    def _subtype_vec(self, x, subtype):
        if not self.use_subtype:
            return None
        if subtype is None:
            subtype = torch.full((x.shape[0],), self.n_subtypes, dtype=torch.long, device=x.device)
        elif self.training and self.subtype_drop > 0:
            drop = torch.rand(subtype.shape[0], device=x.device) < self.subtype_drop
            subtype = torch.where(drop, torch.full_like(subtype, self.n_subtypes), subtype)
        return self.subtype_emb(subtype)

    def _run(self, x, subtype=None):
        zs = torch.stack([e(x.index_select(1, getattr(self, f"idx_{k}")))
                          for k, e in zip(self.group_names, self.experts)], dim=1)  # B,E,d
        s = self._subtype_vec(x, subtype)
        g_logits = self.gate(x if s is None else torch.cat([x, s], 1)) / self.gate_temp
        if self.training and self.expert_dropout > 0 and self.n_experts > 1:
            keep = torch.rand_like(g_logits) >= self.expert_dropout
            # always keep at least one expert (the gate's favourite) per nucleus
            keep[torch.arange(keep.shape[0]), g_logits.argmax(1)] |= ~keep.any(1)
            g_logits = g_logits.masked_fill(~keep, float("-inf"))
        g = torch.softmax(g_logits, dim=1)                                     # B,E
        fused = (g.unsqueeze(-1) * zs).sum(1)                                   # B,d
        return zs, g, fused

    # ----------------------------------------------------------- public API
    def forward(self, x, subtype=None, return_aux=False):
        zs, g, fused = self._run(x, subtype)
        logits = self.head(fused)
        if not return_aux:
            return logits
        expert_logits = torch.stack([h(zs[:, i]) for i, h in enumerate(self.aux_heads)], 1)
        return logits, {"gate": g, "fused": fused, "expert_logits": expert_logits}

    def features(self, x, subtype=None):
        """Fused embedding - what --domain_adversarial attaches to."""
        return self._run(x, subtype)[2]

    def classify(self, feats):
        return self.head(feats)


def auxiliary_loss(aux, y, aux_weight=0.2, balance_weight=0.01, bag=False):
    """Extra MoE terms added to the main CE. Returns (loss, parts dict)."""
    el = aux["expert_logits"]                     # B,E,C
    B, E, C = el.shape
    ce = F.cross_entropy(el.reshape(B * E, C), y.repeat_interleave(E), reduction="none").view(B, E)
    if bag:   # online bagging: each expert gets its own Poisson(1) bootstrap weights
        w = torch.poisson(torch.ones_like(ce))
        aux_ce = (ce * w).sum(0) / w.sum(0).clamp_min(1.0)
    else:
        aux_ce = ce.mean(0)
    aux_term = aux_ce.mean()
    usage = aux["gate"].mean(0).clamp_min(1e-8)   # E
    balance = (usage * (usage * E).log()).sum()   # KL(usage || uniform), >= 0
    loss = aux_weight * aux_term + balance_weight * balance
    return loss, {"aux_ce": aux_term.item(), "balance": balance.item(),
                  "usage": usage.detach().cpu().numpy()}


# -------------------------------------------------------------- diagnostics
def yule_q(ci, cj):
    """Windeatt eq. (6): Q = (N11 N00 - N01 N10) / (N11 N00 + N01 N10), where
    N^ab counts nuclei with expert i correct=a and expert j correct=b.
    +1 = experts err on the same nuclei (no gain from combining); 0 =
    independent errors; negative = complementary errors."""
    n11 = np.sum(ci & cj); n00 = np.sum(~ci & ~cj)
    n10 = np.sum(ci & ~cj); n01 = np.sum(~ci & cj)
    den = n11 * n00 + n01 * n10
    return float((n11 * n00 - n01 * n10) / den) if den else float("nan")


@torch.no_grad()
def report_experts(model, X_t, st_t, y, subtypes_arr, classes, title="VALIDATION"):
    model.eval()
    logits, aux = model(X_t, st_t, return_aux=True)
    pred = logits.argmax(1).cpu().numpy()
    ep = aux["expert_logits"].argmax(2).cpu().numpy()      # N,E
    g = aux["gate"].cpu().numpy()
    names = model.group_names
    correct = ep == y[:, None]

    print(f"\n=== MoE EXPERT DIAGNOSTICS ({title}) ===")
    print(f"fused model acc={np.mean(pred == y):.4f}")
    print("per-expert aux-head accuracy (each expert alone):")
    for i, n in enumerate(names):
        print(f"  {n:<10} acc={correct[:, i].mean():.4f}   mean gate={g[:, i].mean():.3f}")
    oracle = correct.any(1).mean()
    print(f"oracle (any expert right): {oracle:.4f}  - headroom a perfect gate could reach")

    if len(names) > 1:
        print("\npairwise Yule's Q between experts (lower = more diverse errors):")
        print(" " * 11 + "".join(f"{n[:9]:>10}" for n in names))
        qs = []
        for i, a in enumerate(names):
            row = []
            for j, b in enumerate(names):
                if i == j:
                    row.append("      -   ")
                else:
                    q = yule_q(correct[:, i], correct[:, j]); row.append(f"{q:10.3f}")
                    if j > i: qs.append(q)
            print(f"  {a:<9}" + "".join(row))
        print(f"  mean Q = {np.nanmean(qs):.3f}")

    print("\nmean gate weight by stain x true class:")
    rows = []
    for st in sorted(set(subtypes_arr)):
        for ci, c in enumerate(classes):
            m = (subtypes_arr == st) & (y == ci)
            if m.any():
                rows.append((f"{st}/{c}", int(m.sum()), g[m].mean(0)))
    print(f"  {'group':<16}{'n':>8}" + "".join(f"{n[:9]:>10}" for n in names))
    for k, n, v in rows:
        print(f"  {k:<16}{n:>8}" + "".join(f"{x:10.3f}" for x in v))
    return {"expert_acc": dict(zip(names, correct.mean(0).tolist())),
            "gate_mean": dict(zip(names, g.mean(0).tolist())), "oracle": float(oracle)}


# ------------------------------------------------------- build / load helper
def build_from_config(cfg, n_features, n_classes):
    groups = OrderedDict((k, v) for k, v in cfg["groups"])
    return MoEClassifier(n_features, n_classes, groups, cfg["n_subtypes"], d=cfg["d"],
                         expert_hidden=cfg["expert_hidden"], gate_hidden=cfg["gate_hidden"],
                         head_hidden=cfg["head_hidden"], dropout=cfg["dropout"],
                         subtype_dim=cfg["subtype_dim"], use_subtype=cfg["use_subtype"],
                         gate_temp=cfg["gate_temp"], expert_dropout=cfg["expert_dropout"],
                         subtype_drop=cfg["subtype_drop"])


def subtype_indices(subtypes_arr, vocab):
    """Stain names -> embedding indices; anything not in vocab -> 'unknown'."""
    lut = {s: i for i, s in enumerate(vocab)}
    return np.array([lut.get(s, len(vocab)) for s in subtypes_arr], dtype=np.int64)


def load_model(ckpt, device="cpu"):
    """For the deployment/inference script: rebuilds either architecture from
    a train_deploy.py checkpoint. MoE usage at inference:
        model, info = load_model(torch.load(path, weights_only=False))
        df = add_derived_features(df)            # before picking feature_cols
        st = torch.as_tensor(subtype_indices([slide_subtype]*len(X), info["subtype_vocab"]))
        logits = model(X_scaled, st)             # or model(X_scaled) -> 'unknown' stain
    """
    n_f, n_c = len(ckpt["feature_cols"]), len(ckpt["classes"])
    if ckpt.get("arch", "mlp") == "moe":
        model = build_from_config(ckpt["moe_config"], n_f, n_c)
    else:
        from train_deploy import MLP
        model = MLP(n_f, n_c, tuple(ckpt["hidden"]), ckpt["dropout"])
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), {"subtype_vocab": ckpt.get("subtype_vocab", [])}