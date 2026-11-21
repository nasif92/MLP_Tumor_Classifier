#!/usr/bin/env python3
"""Rank every run in a runs/ folder: which configuration gives the best result?

Reads runs/<config>_s<seed>.log, takes the EXTERNAL TEST section (the reference
slides, scored once), and averages over seeds per configuration.

  python rank_results.py runs
  python rank_results.py runs --noise 0.015 --watch FHS24-019049 RDS24-012921 --csv runs/ranking

Per-stain runs named <prefix>_only_<ER|PR|Ki67><suffix> are merged automatically into one
configuration <prefix>_per_stain<suffix> (seed by seed), so they can be ranked against the
pooled-training models.

Main score = mean of the three per-stain slide-mean accuracies (ER, PR, Ki67). Configs that do not
have all three stains are listed separately. Differences under --noise (default 0.015) are
treated as ties.
"""
import argparse
import csv
import glob
import os
import re
import statistics as stats
from collections import defaultdict

STAINS = ["ER", "PR", "Ki67"]
RUN_RE = re.compile(r"^(?P<name>.+)_s(?P<seed>\d+)\.log$")
ONLY_RE = re.compile(r"^(?P<pre>.*)_only_(?P<st>ER|PR|Ki67)(?P<suf>.*)$")


def parse_log(path):
    txt = open(path, errors="replace").read()
    i = txt.find("EXTERNAL TEST")
    if i < 0:
        return None
    ext = txt[i:]
    def grab(label):
        m = re.search(r"^" + re.escape(label) + r":\s*(.+)$", ext, re.M)
        return {k: float(v) for k, v in re.findall(r"(\w+)=([0-9.]+)", m.group(1))} if m else None

    out = {"acc": grab("slide-mean acc by subtype"), "pooled_acc": None, "pooled_f1": None,
           "slides": {}, "epoch": None,
           # present only for runs trained with the Ghost class (train_deploy.report_ghost_aware)
           "forced": grab("forced slide-mean acc by subtype"),
           "strict": grab("strict slide-mean acc by subtype"),
           "kept": grab("kept slide-mean acc by subtype"),
           "cov": grab("coverage slide-mean by subtype"),
           "gP": grab("ghost precision by subtype"),
           "gR": grab("ghost recall by subtype")}
    if out["acc"] is None:
        return None
    for k, a, f in re.findall(r"EXTERNAL TEST\s*-\s*(\w+)[^\n]*\n\s*acc=([0-9.]+)\s+macro_f1=([0-9.]+)", ext):
        # a 3-class pooled accuracy is not comparable with the 2-class runs - leave it out
        if k.upper() == "POOLED" and out["forced"] is None:
            out["pooled_acc"], out["pooled_f1"] = float(a), float(f)
    for slide, _st, _n, a in re.findall(r"^(\S+:\S+)\s+(\S+)\s+(\d+)\s+([0-9.]+)\s*$", ext, re.M):
        out["slides"][slide] = float(a)
    # Ghost runs: the plain per-slide table above scores the 3-class problem (true ghosts included),
    # which is NOT comparable with the 2-class runs. The ghost-aware table has per-slide forced /
    # strict / kept accuracy over the real nuclei only.
    ga = {"forced": {}, "strict": {}, "kept": {}}
    for line in ext.splitlines():
        t = line.split()
        if len(t) == 10 and ":" in t[0] and t[2].isdigit() and t[3].isdigit():
            for key, idx in (("forced", 4), ("strict", 5), ("kept", 7)):
                try:
                    v = float(t[idx])
                except ValueError:
                    continue
                if v == v:
                    ga[key][t[0]] = v
    out["slides_ga"] = ga
    e = re.search(r"Saved best-epoch model \(epoch (\d+)\)", txt)
    if e:
        out["epoch"] = int(e.group(1))
    return out


def ms(v):
    if not v:
        return "-"
    sd = stats.pstdev(v) if len(v) > 1 else 0.0
    return f"{stats.mean(v):.3f}±{sd:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir")
    ap.add_argument("--noise", type=float, default=0.015, help="differences below this count as ties")
    ap.add_argument("--min_seeds", type=int, default=3, help="configs with fewer seeds are flagged")
    ap.add_argument("--watch", nargs="*", default=[], help="slide-name substrings to report per config")
    ap.add_argument("--csv", default=None, help="write <csv>.csv")
    ap.add_argument("--metric", choices=["forced", "strict", "kept", "plain"], default="forced",
                    help="accuracy to rank by. Runs trained WITHOUT the Ghost class have only one "
                         "accuracy and use it for every choice. For Ghost runs: forced = pick "
                         "Tumor/Non Tumor ignoring the Ghost output (comparable to the 2-class runs); "
                         "strict = a Ghost prediction counts as an error; kept = accuracy on the "
                         "nuclei not rejected (see the cov column for how many that is); plain = "
                         "the 3-class accuracy including true ghosts (NOT comparable).")
    a = ap.parse_args()

    raw = defaultdict(dict)  # name -> seed -> parsed
    skipped = []
    for f in sorted(glob.glob(os.path.join(a.runs_dir, "*.log"))):
        m = RUN_RE.match(os.path.basename(f))
        if not m:
            continue
        p = parse_log(f)
        if p is None:
            skipped.append(os.path.basename(f))
            continue
        raw[m["name"]][int(m["seed"])] = p

    # merge per-stain runs: <pre>_only_<ST><suf> -> <pre>_per_stain<suf>, seed by seed
    cfg = defaultdict(dict)  # name -> seed -> {"acc":{}, ...}
    new = lambda: {"acc": {}, "m": {"forced": {}, "strict": {}, "kept": {}}, "cov": {}, "gP": {}, "gR": {}, "slides": {}, "epochs": [],
                   "pooled_acc": None, "pooled_f1": None}
    for name, seeds in raw.items():
        om = ONLY_RE.match(name)
        for seed, p in seeds.items():
            acc = p["acc"] if a.metric == "plain" or not p.get(a.metric) else p[a.metric]
            if om:
                key = f"{om['pre']}_per_stain{om['suf']}"
                d = cfg[key].setdefault(seed, new())
                d["acc"][om["st"]] = acc.get(om["st"])
                for k in d["m"]:
                    if p.get(k):
                        d["m"][k][om["st"]] = p[k].get(om["st"])
                for k in ("cov", "gP", "gR"):
                    if p.get(k):
                        d[k][om["st"]] = p[k].get(om["st"])
            else:
                key = name
                d = cfg[key].setdefault(seed, new())
                d["acc"].update(acc)
                for k in d["m"]:
                    d["m"][k].update(p.get(k) or {})
                for k in ("cov", "gP", "gR"):
                    d[k].update(p.get(k) or {})
                d["pooled_acc"], d["pooled_f1"] = p["pooled_acc"], p["pooled_f1"]
            sl = p["slides_ga"].get(a.metric) if a.metric != "plain" and p["slides_ga"].get(a.metric) else p["slides"]
            d["slides"].update(sl)
            if p["epoch"] is not None:
                d["epochs"].append(p["epoch"])

    rows = []
    for name, seeds in cfg.items():
        per = {s: [d["acc"][s] for d in seeds.values() if d["acc"].get(s) is not None] for s in STAINS}
        mean3 = [stats.mean(d["acc"][s] for s in STAINS) for d in seeds.values()
                 if all(d["acc"].get(s) is not None for s in STAINS)]
        pooled = [d["pooled_acc"] for d in seeds.values() if d["pooled_acc"] is not None]
        f1 = [d["pooled_f1"] for d in seeds.values() if d["pooled_f1"] is not None]
        eps = [e for d in seeds.values() for e in d["epochs"]]
        gh = {k: [stats.mean(v for v in d[k].values() if v is not None)
                  for d in seeds.values() if any(v is not None for v in d[k].values())]
              for k in ("cov", "gP", "gR")}
        mm = {k: [stats.mean(d["m"][k][s] for s in STAINS) for d in seeds.values()
                  if all(d["m"][k].get(s) is not None for s in STAINS)] for k in ("forced", "strict", "kept")}
        mmst = {k: {s: [d["m"][k][s] for d in seeds.values() if d["m"][k].get(s) is not None] for s in STAINS}
                for k in ("forced", "strict", "kept")}
        watch = {}
        for w in a.watch:
            vals = [v for d in seeds.values() for k, v in d["slides"].items() if w in k]
            watch[w] = stats.mean(vals) if vals else None
        rows.append({"name": name, "n": len(seeds), "per": per, "mean3": mean3,
                     "pooled": pooled, "f1": f1, "max_ep": max(eps) if eps else None, "watch": watch,
                     "gh": gh, "mm": mm, "mmst": mmst})

    full = sorted([r for r in rows if r["mean3"]], key=lambda r: -stats.mean(r["mean3"]))
    part = [r for r in rows if not r["mean3"]]
    if not full:
        print("No configuration has all three stains. Nothing to rank.")
        return
    best = stats.mean(full[0]["mean3"])

    has_ghost = any(r["gh"]["cov"] for r in rows)
    hdr = (f"{'config':34s} {'n':>2s} {'mean3':>13s} {'d_best':>7s}  {'ER':>13s} {'PR':>13s} {'Ki67':>13s}"
           f" {'pooled':>13s} {'maxEp':>5s}")
    if has_ghost:
        hdr += f" {'cov':>6s} {'gPrec':>6s} {'gRec':>6s}"
    for w in a.watch:
        hdr += f" {w[:10]:>10s}"
    print(f"\nRanked by mean of the three per-stain slide-mean accuracies, metric = {a.metric} "
          f"(ties: within {a.noise:.3f} of best)")
    if has_ghost:
        print("Ghost-trained runs: cov = fraction of real nuclei not rejected as Ghost; gPrec / gRec = "
              "Ghost precision / recall; '-' = run has no Ghost class.")
    print()
    print(hdr)
    print("-" * len(hdr))
    for r in full:
        m3 = stats.mean(r["mean3"])
        tag = " *" if best - m3 <= a.noise else "  "
        low = " (few seeds)" if r["n"] < a.min_seeds else ""
        line = (f"{r['name'][:34]:34s} {r['n']:2d} {ms(r['mean3']):>13s} {m3 - best:+7.3f}  "
                f"{ms(r['per']['ER']):>13s} {ms(r['per']['PR']):>13s} {ms(r['per']['Ki67']):>13s} "
                f"{ms(r['pooled']):>13s} {str(r['max_ep'] or '-'):>5s}")
        if has_ghost:
            g1 = lambda k: f"{stats.mean(r['gh'][k]):.3f}" if r["gh"][k] else "-"
            line += f" {g1('cov'):>6s} {g1('gP'):>6s} {g1('gR'):>6s}"
        for w in a.watch:
            v = r["watch"][w]
            line += f" {('-' if v is None else f'{v:.3f}'):>10s}"
        print(line + tag + low)
    print("\n* = within the noise band of the best configuration (not distinguishable).")

    # ---- Ghost-trained vs the same configuration trained without Ghost -------------------
    # Accuracies are over REAL nuclei only (true ghosts dropped from the evaluation).
    byname = {r["name"]: r for r in rows}
    pairs = [(r, byname[r["name"][3:]]) for r in rows
             if r["name"].startswith("g3_") and r["name"][3:] in byname and r["mm"]["forced"]]
    if pairs:
        mn = lambda v: stats.mean(v) if v else float("nan")
        print("\nGhost-trained vs the same config trained without Ghost. Accuracy on REAL nuclei only "
              "(true ghosts dropped), mean of the 3 stains.")
        print("  forced = Ghost output ignored | strict = a Ghost prediction counts as an error | "
              "kept = accuracy on the nuclei not rejected as Ghost")
        h = (f"{'config':26s} {'n':>2s} {'no-Ghost':>9s} {'forced':>8s} {'d':>6s} {'strict':>8s} {'d':>6s} "
             f"{'kept':>8s} {'cov':>6s} {'gPrec':>6s} {'gRec':>6s}")
        print(h); print("-" * len(h))
        for g, b in sorted(pairs, key=lambda x: -mn(x[0]["mm"]["forced"])):
            base = mn(b["mean3"] if a.metric == "plain" else b["mean3"])
            f_, s_, k_ = (mn(g["mm"][k]) for k in ("forced", "strict", "kept"))
            print(f"{g['name'][3:][:26]:26s} {g['n']:2d} {base:9.3f} {f_:8.3f} {f_ - base:+6.3f} "
                  f"{s_:8.3f} {s_ - base:+6.3f} {k_:8.3f} "
                  f"{mn(g['gh']['cov']):6.3f} {mn(g['gh']['gP']):6.3f} {mn(g['gh']['gR']):6.3f}")
        print("\nPer stain, forced accuracy: no-Ghost -> Ghost-trained")
        h2 = f"{'config':26s} {'ER':>17s} {'PR':>17s} {'Ki67':>17s}"
        print(h2); print("-" * len(h2))
        for g, b in sorted(pairs, key=lambda x: -mn(x[0]["mm"]["forced"])):
            cells = []
            for s in STAINS:
                bb, gg = mn(b["per"][s]), mn(g["mmst"]["forced"][s])
                cells.append(f"{bb:.3f} -> {gg:.3f}")
            print(f"{g['name'][3:][:26]:26s} " + " ".join(f"{c:>17s}" for c in cells))

    print("\nBest configuration per stain (configs with >= min_seeds seeds):")
    for s in STAINS:
        c = [(stats.mean(r["per"][s]), r["name"], r["n"]) for r in rows if r["per"][s] and r["n"] >= a.min_seeds]
        c.sort(reverse=True)
        if c:
            top = c[0][0]
            ties = [n for v, n, _ in c if top - v <= a.noise]
            print(f"  {s:5s} best {c[0][1]} = {top:.3f};  {len(ties)} config(s) within noise")

    if part:
        print("\nConfigurations without all three stains (not ranked):")
        for r in part:
            print(f"  {r['name']:34s} n={r['n']}  " + "  ".join(f"{s}={ms(r['per'][s])}" for s in STAINS if r["per"][s]))
    if skipped:
        print(f"\nSkipped (no external-test section found, probably unfinished): {len(skipped)} log(s)")
        print("  " + ", ".join(skipped[:10]) + (" ..." if len(skipped) > 10 else ""))

    if a.csv:
        with open(a.csv + ".csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["config", "n_seeds", "mean3", "sd3", "ER", "PR", "Ki67", "pooled", "pooled_f1", "max_best_epoch",
                        "coverage", "ghost_precision", "ghost_recall"] + [f"slide:{x}" for x in a.watch])
            for r in full + part:
                mean = lambda v: round(stats.mean(v), 4) if v else ""
                w.writerow([r["name"], r["n"], mean(r["mean3"]),
                            round(stats.pstdev(r["mean3"]), 4) if len(r["mean3"]) > 1 else "",
                            mean(r["per"]["ER"]), mean(r["per"]["PR"]), mean(r["per"]["Ki67"]),
                            mean(r["pooled"]), mean(r["f1"]), r["max_ep"] or "",
                            mean(r["gh"]["cov"]), mean(r["gh"]["gP"]), mean(r["gh"]["gR"])]
                           + [("" if r["watch"][x] is None else round(r["watch"][x], 4)) for x in a.watch])
        print(f"\nWrote {a.csv}.csv")


if __name__ == "__main__":
    main()