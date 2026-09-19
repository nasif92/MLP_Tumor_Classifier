"""
Generic comparison of two GeoJSON detection/classification files for the
SAME slide - by classification.name, matched by nearest centroid. Works for
any two sources: two different models' output, a model vs QuPath's RF
classifier, two different QuPath projects, etc. Neither input is treated as
ground truth - this just reports how often they agree.

If a file has more than the classes you care about (e.g. QuPath's RF export
uses raw names like "Tumor"/"Stroma"/"Immune cells"/"Ignore*"), pass
--collapse to run each side's class name through config.collapse_label()
first, which maps everything down to Tumor/Non Tumor/Normal (or None -
dropped) - the same collapsing used everywhere else in this pipeline. Leave
--collapse off to compare raw class names as-is instead.

Usage:
    # Two model outputs, both already Tumor/Non Tumor (raw comparison):
    python compare_geojsons.py --geojson_a slideX_pooled.geojson \\
        --geojson_b slideX_single.geojson --label_a pooled --label_b single

    # QuPath RF export (raw multi-class labels) vs a model's Tumor/Non Tumor
    # output - collapse the RF side down to the same 2 classes first:
    python compare_geojsons.py --geojson_a qupath_rf_detections.geojson.gz \\
        --geojson_b slideX_pooled.geojson --label_a rf --label_b pooled --collapse

    # Save an annotated copy (geometry from --geojson_a) with a match/
    # mismatch classification for QuPath:
    python compare_geojsons.py --geojson_a a.geojson --geojson_b b.geojson \\
        --label_a A --label_b B --out_geojson compared.geojson
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_features import load_geojson  # noqa: E402
from config import collapse_label  # noqa: E402

try:
    from scipy.spatial import cKDTree
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

PALETTE = [
    [46, 139, 87], [220, 20, 60], [70, 130, 180], [255, 140, 0],
    [148, 0, 211], [255, 215, 0], [0, 139, 139], [199, 21, 133],
]


def feature_centroid(geom):
    t = geom.get("type")
    coords = geom.get("coordinates")
    if t == "Point":
        return coords[0], coords[1]
    if t == "Polygon":
        ring = coords[0]
        xs = [p[0] for p in ring]; ys = [p[1] for p in ring]
        return sum(xs) / len(xs), sum(ys) / len(ys)
    if t == "MultiPolygon":
        ring = coords[0][0]
        xs = [p[0] for p in ring]; ys = [p[1] for p in ring]
        return sum(xs) / len(xs), sum(ys) / len(ys)
    return None


def get_feature_id(feat, id_property):
    """Same convention as export_geojson.py's get_feature_id: try
    (in order) an explicit --id_property, else nucleus_id/id/objectID/name,
    checked under properties first and at the feature's top level second
    (GeoJSON allows a top-level "id" member on a Feature - that's how
    QuPath commonly persists a stable object id across different
    classification exports of the SAME underlying detections)."""
    props = feat.get("properties", {}) or {}
    keys_to_try = [id_property] if id_property else ["nucleus_id", "id", "objectID", "name"]
    for k in keys_to_try:
        if k is None:
            continue
        if k in props:
            return str(props[k])
        if k in feat:
            return str(feat[k])
    return None


def get_class_name(feat, class_property):
    """Default: QuPath's standard properties.classification.name. Falls
    back to a couple of other common shapes, or a fully custom dotted
    --class_property path if given (e.g. 'properties.pooled_model_pred' for
    a flat string property instead of the nested classification dict)."""
    props = feat.get("properties", {}) or {}
    if class_property:
        cur = feat
        for part in class_property.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return None
        return cur if isinstance(cur, str) else None
    cls = props.get("classification")
    if isinstance(cls, dict) and cls.get("name"):
        return cls["name"]
    if isinstance(props.get("class"), str):
        return props["class"]
    if isinstance(props.get("name"), str):
        return props["name"]
    return None


_COLLAPSED_TARGETS = {"Tumor", "Non Tumor", "Normal"}

# Spelling/casing variants seen in the wild across different tools' exports
# that all mean one of the 3 collapsed target labels, but aren't recognised
# as valid input by config.collapse_label() (which only maps ITS OWN raw
# annotation vocabulary - Stroma, Immune cells, Text*, etc. - to these
# targets, not other tools' spelling of the targets themselves). Checked
# case-insensitively so "NON-Tumor", "non-tumor", "NON TUMOR" etc. all
# normalize the same way, BEFORE collapse_label ever sees the value -
# without this, a value like QuPath's "NON-Tumor" doesn't match "Non Tumor"
# literally, isn't recognised by collapse_label either, and gets silently
# dropped from the comparison entirely (looks like a class the two files
# don't share, when it's really just a spelling difference).
_LABEL_ALIASES = {
    "non-tumor": "Non Tumor",
    "non tumor": "Non Tumor",
    "nontumor": "Non Tumor",
    "tumor": "Tumor",
    "normal": "Normal",
}


def safe_collapse(raw):
    """collapse_label() maps various raw annotation names TO 'Non Tumor'
    etc., but doesn't recognise those target names as already-valid input -
    so collapsing an already-collapsed value (e.g. a model's own 'Non
    Tumor' output) would silently drop it. Pass already-valid values
    through unchanged instead of re-running them through collapse_label().
    Also normalizes known spelling/casing variants of the target labels
    (see _LABEL_ALIASES) before falling through to collapse_label() for
    anything else."""
    if raw in _COLLAPSED_TARGETS:
        return raw
    alias = _LABEL_ALIASES.get(raw.strip().lower()) if isinstance(raw, str) else None
    if alias is not None:
        return alias
    return collapse_label(raw)


def _cache_path(cache_dir, path, class_property, collapse, id_property, exclude_classes):
    """Cache key covers everything that changes the extracted result: the
    source file's path/size/mtime (so an edited or replaced file invalidates
    the cache automatically) plus every option that affects what gets
    extracted from it."""
    try:
        st = os.stat(path)
        sig = (f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}|"
               f"{class_property}|{collapse}|{id_property}|"
               f"{','.join(sorted(exclude_classes)) if exclude_classes else ''}")
    except OSError:
        return None
    key = hashlib.md5(sig.encode()).hexdigest()
    return os.path.join(cache_dir, f"{key}.npz")


def load_labeled_points(path, class_property, collapse, id_property,
                         cache_dir=None, need_features=True, exclude_classes=None):
    """need_features=False (the common case - no --out_geojson requested)
    lets this be served from a cache of just the extracted points/labels/ids
    instead of re-parsing and re-walking the whole GeoJSON every run - the
    win that actually matters at 1M+ features/file, since re-parsing the
    same file for a second comparison (different --tolerance_px, --classes,
    etc.) is otherwise just as slow as the first time. Caching is skipped
    entirely when --out_geojson needs the original Feature objects for
    output geometry/properties - that always does a fresh parse.

    exclude_classes drops features by their RAW class name (before
    --collapse) before they're ever turned into a centroid or added to the
    arrays that get matched - e.g. QuPath's tiling-grid "tile" objects,
    which have area=0 and aren't real nucleus detections at all. Filtering
    these out here (not just at --classes time, at the very end) is what
    actually shrinks the cKDTree / id-matching workload at 1M+ features,
    since otherwise every tile still gets a centroid computed and gets
    built into the tree even though it can never be a real match."""
    exclude_classes = set(exclude_classes or [])
    cpath = _cache_path(cache_dir, path, class_property, collapse, id_property, exclude_classes) \
        if (cache_dir and not need_features) else None
    if cpath and os.path.exists(cpath):
        d = np.load(cpath, allow_pickle=True)
        pts, labels, ids = d["pts"], d["labels"], list(d["ids"])
        print(f"  {path}: loaded {len(pts)} usable features from cache "
              f"({os.path.basename(cpath)}) - skipped re-parsing the source file")
        return pts, labels, ids, []

    features = load_geojson(path)
    pts, labels, ids, kept_features = [], [], [], []
    n_no_class, n_no_geom, n_dropped_by_collapse, n_excluded = 0, 0, 0, 0
    for feat in features:
        raw = get_class_name(feat, class_property)
        if raw is None:
            n_no_class += 1
            continue
        if raw in exclude_classes:
            n_excluded += 1
            continue
        label = safe_collapse(raw) if collapse else raw
        if label is None:
            n_dropped_by_collapse += 1
            continue
        geom = feat.get("geometry")
        c = feature_centroid(geom) if geom else None
        if c is None:
            n_no_geom += 1
            continue
        pts.append(c)
        labels.append(label)
        ids.append(get_feature_id(feat, id_property))
        if need_features:
            kept_features.append(feat)
    print(f"  {path}: {len(features)} features | {len(pts)} usable "
          f"(no classification={n_no_class}, excluded ({sorted(exclude_classes)})={n_excluded}, "
          f"no geometry={n_no_geom}, dropped by collapse_label={n_dropped_by_collapse})")

    pts_arr = np.asarray(pts, dtype=float)
    labels_arr = np.array(labels, dtype=object)
    if cpath:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(cpath, pts=pts_arr, labels=labels_arr,
                            ids=np.array(ids, dtype=object))
        print(f"  cached extracted result -> {cpath} (future runs against this "
              f"exact file + options skip re-parsing)")
    return pts_arr, labels_arr, ids, kept_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geojson_a", required=True)
    ap.add_argument("--geojson_b", required=True)
    ap.add_argument("--label_a", default="A", help="Name for --geojson_a in reports.")
    ap.add_argument("--label_b", default="B", help="Name for --geojson_b in reports.")
    ap.add_argument("--collapse_a", action="store_true",
                     help="Run --geojson_a's raw classification names through "
                          "config.collapse_label() before comparing (maps raw "
                          "annotation-style names like 'Stroma'/'Immune cells' "
                          "down to Tumor/Non Tumor/Normal, dropping anything "
                          "else e.g. 'Ignore*'). Use this on whichever side "
                          "actually has raw multi-class names - a side that's "
                          "already Tumor/Non Tumor should NOT be collapsed "
                          "again, since collapse_label() maps things TO "
                          "'Non Tumor' but doesn't recognise 'Non Tumor' "
                          "itself as already-valid input and would drop it.")
    ap.add_argument("--collapse_b", action="store_true",
                     help="Same as --collapse_a, for --geojson_b.")
    ap.add_argument("--classes", nargs="+", default=None,
                     help="Restrict comparison to these class names (both "
                          "sides). Defaults to every class name that appears "
                          "on both sides.")
    ap.add_argument("--class_property_a", default=None,
                     help="Dotted property path for --geojson_a's class name "
                          "if not the standard properties.classification.name "
                          "(e.g. 'properties.pooled_model_pred').")
    ap.add_argument("--class_property_b", default=None,
                     help="Same as --class_property_a, for --geojson_b.")
    ap.add_argument("--id_property", default=None,
                     help="Property name holding a stable per-nucleus id shared "
                          "by BOTH files (e.g. QuPath's own object id, persisted "
                          "across different classification exports of the same "
                          "underlying detections). If omitted, tries nucleus_id/"
                          "id/objectID/name (checked under properties, then at "
                          "the feature's top level). Features with a matching id "
                          "on both sides are paired directly (fast, exact) - "
                          "nearest-centroid matching is only used as a fallback "
                          "for whatever's left unmatched by id (or everything, "
                          "if neither file has a recognisable id at all).")
    ap.add_argument("--exclude_class", nargs="+", default=["tile"],
                     help="Drop features whose RAW class name (before "
                          "--collapse) is in this list, before centroid/id "
                          "matching - not just at the final --classes filter. "
                          "Defaults to ['tile'], QuPath's tiling-grid objects "
                          "(area=0, not real nucleus detections) that would "
                          "otherwise get built into the matching index for no "
                          "reason. Pass --exclude_class (with nothing after "
                          "it) to disable filtering entirely.")
    ap.add_argument("--no_id_match", action="store_true",
                     help="Skip id-based matching entirely and always use "
                          "nearest-centroid matching, even if both files "
                          "happen to share an id property.")
    ap.add_argument("--tolerance_px", type=float, default=5.0,
                     help="Max centroid distance for matching a feature in A "
                          "to its nearest feature in B (used for whatever "
                          "wasn't already matched by id).")
    ap.add_argument("--cache_dir", default=".geojson_compare_cache",
                     help="Where to cache each file's extracted points/labels/"
                          "ids, keyed by file path+size+mtime+options - a "
                          "second run against the same file (e.g. trying a "
                          "different --tolerance_px or --classes) loads from "
                          "here instead of re-parsing a huge GeoJSON from "
                          "scratch. Skipped automatically when --out_geojson "
                          "is given (that needs a fresh parse for geometry). "
                          "Set to '' to disable caching entirely.")
    ap.add_argument("--out_geojson", default=None,
                     help="If given, writes a new geojson (geometry from "
                          "--geojson_a) with each matched feature's "
                          "properties augmented with label_a/label_b's class "
                          "names and a '<A's class>-match'/'-mismatch' "
                          "classification. Neither input file is modified.")
    args = ap.parse_args()

    cache_dir = args.cache_dir or None
    need_features = bool(args.out_geojson)  # only the --out_geojson path needs raw Features

    print(f"Loading --geojson_a ({args.label_a}):")
    pts_a, labels_a, ids_a, feats_a = load_labeled_points(
        args.geojson_a, args.class_property_a, args.collapse_a, args.id_property,
        cache_dir=cache_dir, need_features=need_features, exclude_classes=args.exclude_class)
    print(f"Loading --geojson_b ({args.label_b}):")
    pts_b, labels_b, ids_b, feats_b = load_labeled_points(
        args.geojson_b, args.class_property_b, args.collapse_b, args.id_property,
        cache_dir=cache_dir, need_features=need_features, exclude_classes=args.exclude_class)

    if len(pts_a) == 0 or len(pts_b) == 0:
        raise SystemExit("One of the two files had no usable (classified + geometry) "
                         "features - nothing to compare.")

    n_a = len(pts_a)
    idx = np.full(n_a, -1, dtype=int)
    dist = np.full(n_a, np.inf)
    matched_via_id = np.zeros(n_a, dtype=bool)

    if not args.no_id_match:
        # Fast, exact path: pair by a shared id where both sides have one.
        # Last-one-wins on duplicate ids in B (shouldn't happen for a real
        # per-nucleus id, but doesn't crash if it does).
        b_id_to_idx = {bid: j for j, bid in enumerate(ids_b) if bid is not None}
        if b_id_to_idx:
            for i, aid in enumerate(ids_a):
                if aid is not None and aid in b_id_to_idx:
                    idx[i] = b_id_to_idx[aid]
                    dist[i] = 0.0
                    matched_via_id[i] = True
        n_id_matched = int(matched_via_id.sum())
        if n_id_matched:
            print(f"\nid-based matching: {n_id_matched}/{n_a} of A's features "
                  f"paired directly via a shared id property "
                  f"(property tried: {args.id_property or 'nucleus_id/id/objectID/name'})")
        else:
            print(f"\nid-based matching: no shared id found between the two files "
                  f"(tried: {args.id_property or 'nucleus_id/id/objectID/name'}) - "
                  f"falling back to nearest-centroid matching for everything.")

    remaining = np.where(~matched_via_id)[0]
    if len(remaining):
        if HAVE_SCIPY:
            tree = cKDTree(pts_b)
            d_rem, idx_rem = tree.query(pts_a[remaining])
        else:
            print("  NOTE: scipy not found - using slower brute-force matching.")
            d_rem = np.empty(len(remaining)); idx_rem = np.empty(len(remaining), dtype=int)
            for k, i in enumerate(remaining):
                d = np.hypot(pts_b[:, 0] - pts_a[i, 0], pts_b[:, 1] - pts_a[i, 1])
                j = int(np.argmin(d))
                d_rem[k], idx_rem[k] = d[j], j
        dist[remaining] = d_rem
        idx[remaining] = idx_rem

    matched = (dist <= args.tolerance_px) & (idx >= 0)
    n_matched, n_unmatched = int(matched.sum()), int((~matched).sum())
    n_centroid_matched = int((matched & ~matched_via_id).sum())
    print(f"Matching (tolerance={args.tolerance_px}px): {n_matched} matched total "
          f"({int(matched_via_id.sum())} by id, {n_centroid_matched} by nearest "
          f"centroid), {n_unmatched} of A's features had no B feature within tolerance")

    a_lab = labels_a[matched]
    b_lab = labels_b[idx[matched]]

    classes = args.classes
    if classes is None:
        classes = sorted(set(a_lab.tolist()) & set(b_lab.tolist()))
    keep = np.isin(a_lab, classes) & np.isin(b_lab, classes)
    a_lab, b_lab = a_lab[keep], b_lab[keep]
    a_feats_matched = [f for f, k in zip((feats_a[i] for i in range(len(feats_a)) if matched[i]), keep) if k]
    b_idx_matched = [j for j, k in zip((idx[i] for i in range(len(idx)) if matched[i]), keep) if k]

    print(f"{len(a_lab)} nuclei comparable after restricting to classes {classes}")
    if len(a_lab) == 0:
        raise SystemExit("Nothing comparable - check --classes / --collapse / --tolerance_px.")

    from sklearn.metrics import classification_report, confusion_matrix
    acc = float((a_lab == b_lab).mean())
    print(f"\n=== {args.label_a} vs {args.label_b} ===")
    print(f"agreement = {acc:.4f}")
    print(classification_report(a_lab, b_lab, labels=classes,
                                target_names=[f"{args.label_a}={c}" for c in classes],
                                zero_division=0))
    cm = confusion_matrix(a_lab, b_lab, labels=classes)
    print(f"Confusion (rows={args.label_a}, cols={args.label_b}): " + "  ".join(classes))
    for i, row in enumerate(cm):
        print(f"{classes[i]:>14}: " + " ".join(f"{v:8d}" for v in row))

    if args.out_geojson:
        color_of = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(classes)}
        out_feats = []
        for feat, la, lb in zip(a_feats_matched, a_lab, b_lab):
            props = dict(feat.get("properties", {}) or {})
            match = "match" if la == lb else "mismatch"
            name = f"{la}-{match}"
            props[f"{args.label_a}_label"] = la
            props[f"{args.label_b}_label"] = lb
            props["classification"] = {"name": name, "color": color_of.get(la, [128, 128, 128])}
            out_feats.append({"type": "Feature", "geometry": feat.get("geometry"), "properties": props})
        os.makedirs(os.path.dirname(args.out_geojson) or ".", exist_ok=True)
        with open(args.out_geojson, "w") as f:
            json.dump({"type": "FeatureCollection", "features": out_feats}, f)
        print(f"\nSaved -> {args.out_geojson} (neither --geojson_a nor --geojson_b was modified)")


if __name__ == "__main__":
    main()