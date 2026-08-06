"""
Batch nucleus feature extraction across subtypes.

Only the dataset directory differs between subtypes, so everything lives in
SUBTYPES below and nothing needs editing between runs:

    python batch_extract.py --subtype TNBC
    python batch_extract.py --subtype all          # every subtype
    python batch_extract.py --subtype all --dry_run
    python batch_extract.py --subtype NSCLC --slides B1 B2
    python batch_extract.py --subtype all --overwrite

Completed slides are skipped, so a run can be interrupted and resumed.
"""
import argparse
import glob
import os
import time
import traceback

import numpy as np
import pandas as pd

import extract_features as ef

# ---------------------------------------------------------------------------
# CONFIG - subtype -> dataset directory name. Everything else is templated.
# ---------------------------------------------------------------------------
SUBTYPES = {
    "TNBC":    "TNBC-D",
    "Cervix":  "Cervix-C",
    "HNSCC":   "HNSCC-A",
    "NSCLC":   "NSCLC-B",
    "UpperGI": "UpperGI-E",
}

WSI_PATTERN = "/mnt/NAS/PDL1-2026/{dir}/{slide}.svs"
DET_PATTERN = "/mnt/NAS/PDL1-2026-Detections/{dir}/cellpose-dino/{slide}/{slide}.geojson.gz"
ANN_PATTERN = "/mnt/NAS/NASIF/Nasif-3rd-batch/tiles_multiclass_regions/{slide}/{slide}_regions.geojson"
OUT_PATTERN = "{subtype}/{slide}_gt_measurements.csv"
DISCOVER_GLOB = "/mnt/NAS/PDL1-2026-Detections/{dir}/cellpose-dino/*/*.geojson.gz"


def paths_for(subtype, slide):
    d = SUBTYPES[subtype]
    fmt = dict(dir=d, slide=slide, subtype=subtype)
    return {"wsi": WSI_PATTERN.format(**fmt),
            "det": DET_PATTERN.format(**fmt),
            "ann": ANN_PATTERN.format(**fmt),
            "out": OUT_PATTERN.format(**fmt)}


def discover_slides(subtype):
    names = set()
    for p in glob.glob(DISCOVER_GLOB.format(dir=SUBTYPES[subtype])):
        base = os.path.basename(p)
        for suffix in (".geojson.gz", ".geojson"):
            if base.endswith(suffix):
                names.add(base[: -len(suffix)])
                break
    # Natural sort, so B2 comes before B10.
    def key(s):
        i = len(s.rstrip("0123456789"))
        return (s[:i], int(s[i:]) if s[i:].isdigit() else 0)
    return sorted(names, key=key)


def process_slide(subtype, slide, args):
    p = paths_for(subtype, slide)

    missing = [(k, v) for k, v in
               (("wsi", p["wsi"]), ("detections", p["det"]),
                ("annotations", p["ann"])) if not os.path.exists(v)]
    if missing:
        return {"slide": slide, "status": "missing",
                "detail": ", ".join(k for k, _ in missing)}

    if os.path.exists(p["out"]) and not args.overwrite:
        try:
            return {"slide": slide, "status": "skipped",
                    "n": len(pd.read_csv(p["out"]))}
        except Exception:
            pass  # unreadable - fall through and redo it

    os.makedirs(os.path.dirname(p["out"]) or ".", exist_ok=True)
    t0 = time.time()

    polys = ef.polygons_from_geojson(p["det"])
    if not polys:
        return {"slide": slide, "status": "error", "detail": "no polygons"}

    centroids_all = np.array([q.mean(axis=0) for q in polys])
    # Every detection's position is kept for neighbour counts, even though
    # features are only extracted for annotated nuclei - otherwise
    # nearby_count would measure annotation density, not tissue density.
    centroids_every_detection = centroids_all.copy()

    labels = ef.assign_labels(centroids_all, p["ann"], verbose=False)
    keep = [i for i, l in enumerate(labels) if l is not None]
    if not keep:
        return {"slide": slide, "status": "error",
                "detail": "no nuclei inside any annotation"}

    polys = [polys[i] for i in keep]
    centroids_all = centroids_all[keep]
    labels = [labels[i] for i in keep]

    with ef.NucleusFeatureExtractor(p["wsi"], mpp=args.mpp) as ext:
        base, centroids = ext.extract_many(polys, centroids=centroids_all,
                                           progress_every=0)
        mpp = ext.mpp

    df = ef.add_smoothed_features(base, centroids, mpp,
                                  all_centroids_px=centroids_every_detection)
    df.insert(0, "cy_wsi", centroids[:, 1])
    df.insert(0, "cx_wsi", centroids[:, 0])
    df.insert(0, "nucleus_id", range(len(df)))
    df.insert(0, "wsi_name", slide)
    df["label"] = labels
    df["is_ground_truth"] = True
    df.to_csv(p["out"], index=False)

    return {"slide": slide, "status": "ok", "n": len(df),
            "seconds": time.time() - t0,
            "counts": pd.Series(labels).value_counts().to_dict(),
            "n_detections": len(centroids_every_detection)}


def run_subtype(subtype, args):
    slides = args.slides or discover_slides(subtype)
    if not slides:
        print(f"  no slides found under "
              f"{DISCOVER_GLOB.format(dir=SUBTYPES[subtype])}")
        return []

    print(f"\n{'='*70}\n{subtype}  ({SUBTYPES[subtype]})  -  {len(slides)} slides\n{'='*70}")

    if args.dry_run:
        for s in slides:
            p = paths_for(subtype, s)
            flags = []
            for k in ("wsi", "det", "ann"):
                flags.append(f"{k}:{'ok' if os.path.exists(p[k]) else 'MISSING'}")
            done = "done" if os.path.exists(p["out"]) else "-"
            print(f"  {s:<8} {'  '.join(flags)}   out:{done}")
        return []

    results = []
    for i, slide in enumerate(slides, 1):
        print(f"  [{i}/{len(slides)}] {slide} ... ", end="", flush=True)
        try:
            r = process_slide(subtype, slide, args)
        except Exception as e:
            r = {"slide": slide, "status": "error", "detail": str(e)}
            traceback.print_exc()
        r["subtype"] = subtype
        results.append(r)

        if r["status"] == "ok":
            print(f"{r['n']} nuclei / {r['n_detections']} detections "
                  f"({r['seconds']:.0f}s)")
            print(f"           {r['counts']}")
        elif r["status"] == "skipped":
            print(f"skipped ({r['n']} already extracted)")
        else:
            print(f"{r['status'].upper()}: {r['detail']}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subtype", default="all",
                    help="One of " + ", ".join(SUBTYPES) + ", or 'all'")
    ap.add_argument("--slides", nargs="+", default=None,
                    help="Specific slides (only with a single --subtype)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--mpp", type=float, default=None)
    args = ap.parse_args()

    if args.subtype == "all":
        todo = list(SUBTYPES)
        if args.slides:
            raise SystemExit("--slides needs a single --subtype")
    elif args.subtype in SUBTYPES:
        todo = [args.subtype]
    else:
        raise SystemExit(f"Unknown subtype '{args.subtype}'. "
                         f"Choose from: {', '.join(SUBTYPES)}, all")

    all_results = []
    for st in todo:
        all_results += run_subtype(st, args)

    if args.dry_run or not all_results:
        return

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    grand = {}
    total_nuclei = 0
    print(f"{'subtype':<10} {'ok':>4} {'skip':>5} {'fail':>5} {'nuclei':>9}")
    for st in todo:
        rs = [r for r in all_results if r["subtype"] == st]
        if not rs:
            continue
        ok = [r for r in rs if r["status"] == "ok"]
        sk = [r for r in rs if r["status"] == "skipped"]
        bad = [r for r in rs if r["status"] not in ("ok", "skipped")]
        n = sum(r.get("n", 0) for r in ok + sk)
        total_nuclei += n
        print(f"{st:<10} {len(ok):>4} {len(sk):>5} {len(bad):>5} {n:>9}")
        for r in ok:
            for k, v in r.get("counts", {}).items():
                grand[k] = grand.get(k, 0) + v

    print(f"{'TOTAL':<10} {'':>4} {'':>5} {'':>5} {total_nuclei:>9}")

    if grand:
        print("\nClass totals (newly extracted this run):")
        for k, v in sorted(grand.items(), key=lambda kv: -kv[1]):
            print(f"  {k:<24} {v:>8}")

    bad = [r for r in all_results if r["status"] not in ("ok", "skipped")]
    if bad:
        print(f"\nProblems ({len(bad)}):")
        for r in bad:
            print(f"  {r['subtype']}/{r['slide']}: {r['status']} - {r['detail']}")


if __name__ == "__main__":
    main()