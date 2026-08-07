"""
Nucleus feature extraction - Python version.

Produces the same 50 measurements as the QuPath pipeline, so output is a
drop-in replacement for *_gt_measurements.csv.

--------------------------------------------------------------------------
API (import this module from a segmentation script)
--------------------------------------------------------------------------

    from extract_features import NucleusFeatureExtractor, add_smoothed_features

    ext = NucleusFeatureExtractor("slide.svs")        # or (arr, mpp=0.26)

    # one nucleus at a time
    for poly in polygons:                              # (N,2) array, slide px
        feats = ext.extract_one(poly)                  # dict of 16 base features

    # or a batch
    df, centroids = ext.extract_many(polygons)         # DataFrame + (N,2) array

    # smoothed features need the whole population (they average over
    # neighbours), so they are a separate pass over the base table
    full = add_smoothed_features(df, centroids, ext.mpp)

The 16 base features are independent per nucleus. The remaining 34 are
Gaussian-weighted neighbourhood averages (FWHM 50um and 75um) plus two
neighbour counts, matching QuPath's SmoothFeaturesPlugin.

--------------------------------------------------------------------------
CLI
--------------------------------------------------------------------------
    python extract_features.py --wsi slide.svs --geojson dets.geojson.gz \\
        --out slide_features.csv
"""
import argparse
import gzip
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# DEFAULT PATHS - edit these to run with no command-line arguments.
# Any of them can still be overridden with --wsi / --geojson / --annotations.
# Set DEFAULT_ANNOTATIONS to None to skip label assignment.
# ---------------------------------------------------------------------------
DEFAULT_WSI = "/mnt/NAS/PDL1-2026/TNBC-D/D1.svs"
DEFAULT_GEOJSON = "/mnt/NAS/PDL1-2026-Detections/TNBC-D/cellpose-dino/D1.geojson.gz"
DEFAULT_ANNOTATIONS = "/mnt/NAS/QuPath_Projects_AA/cellpose-dino-cls/annotation_exports/D1/D1_annotations.geojson"
DEFAULT_OUT = "D1_gt_measurements.csv"

# ---------------------------------------------------------------------------
# H-DAB stain vectors - must match the QuPath setColorDeconvolutionStains
# call ("H-DAB default") for intensity features to be comparable.
# ---------------------------------------------------------------------------
HEMATOXYLIN = np.array([0.6511078257574492, 0.7011930431234068, 0.29049426072255424])
DAB = np.array([0.2691668720495607, 0.5682411743268503, 0.7775931859209531])
BACKGROUND = np.array([255.0, 255.0, 255.0])

SHAPE_FEATURES = ["area_um2", "length_um", "circularity", "solidity",
                  "max_diam_um", "min_diam_um"]
# Intensity is measured INSIDE the nucleus boundary (QuPath
# IntensityFeaturesPlugin with region=NUCLEUS), matching the reference
# classifier's "Hematoxylin: Mean" / "DAB: Median" measurements - not the
# 25um square window, which averages in surrounding tissue.
INTENSITY_FEATURES = [f"{s}_{m}" for s in ("hem", "dab")
                      for m in ("mean", "median", "min", "max", "std")]
BASE_FEATURES = SHAPE_FEATURES + INTENSITY_FEATURES

# Maps our short column names onto the QuPath measurement names the
# reference classifier expects, for cross-checking feature parity.
QUPATH_NAME_MAP = {
    "area_um2": "Area µm^2", "length_um": "Length µm",
    "circularity": "Circularity", "solidity": "Solidity",
    "max_diam_um": "Max diameter µm", "min_diam_um": "Min diameter µm",
    "hem_mean": "Hematoxylin: Mean", "hem_median": "Hematoxylin: Median",
    "hem_min": "Hematoxylin: Min", "hem_max": "Hematoxylin: Max",
    "hem_std": "Hematoxylin: Std.Dev.",
    "dab_mean": "DAB: Mean", "dab_median": "DAB: Median",
    "dab_min": "DAB: Min", "dab_max": "DAB: Max", "dab_std": "DAB: Std.Dev.",
}


def _build_stain_matrix():
    """Third vector is the cross product of the first two (QuPath's
    convention for a 2-stain setup), giving an invertible 3x3."""
    h = HEMATOXYLIN / np.linalg.norm(HEMATOXYLIN)
    d = DAB / np.linalg.norm(DAB)
    r = np.cross(h, d)
    n = np.linalg.norm(r)
    r = r / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
    return np.linalg.inv(np.stack([h, d, r]))


STAIN_INV = _build_stain_matrix()


def deconvolve(rgb):
    """RGB uint8 -> (hematoxylin, dab) optical density channels."""
    rgb = np.maximum(rgb.astype(np.float64), 1.0)  # avoid log(0)
    od = -np.log10(rgb / BACKGROUND[None, None, :])
    stains = (od.reshape(-1, 3) @ STAIN_INV).reshape(od.shape)
    return stains[..., 0], stains[..., 1]


# ===========================================================================
# Geometry - pure functions on a polygon in slide pixels
# ===========================================================================

def polygon_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_perimeter(pts):
    d = np.diff(np.vstack([pts, pts[:1]]), axis=0)
    return float(np.sqrt((d ** 2).sum(axis=1)).sum())


def convex_hull_area(pts):
    if len(pts) < 3:
        return polygon_area(pts)
    try:
        from scipy.spatial import ConvexHull
        return float(ConvexHull(pts).volume)  # 2D "volume" is area
    except Exception:
        return polygon_area(pts)


def feret_diameters(pts):
    """Max and min caliper width. Min is taken over hull edge normals -
    the standard rotating-calipers result."""
    if len(pts) < 3:
        if len(pts) < 2:
            return 0.0, 0.0
        return float(np.linalg.norm(pts[1] - pts[0])), 0.0
    try:
        from scipy.spatial import ConvexHull
        hull = pts[ConvexHull(pts).vertices]
    except Exception:
        hull = pts

    diff = hull[:, None, :] - hull[None, :, :]
    max_d = float(np.sqrt((diff ** 2).sum(-1)).max())

    edges = np.roll(hull, -1, axis=0) - hull
    lengths = np.linalg.norm(edges, axis=1)
    keep = lengths > 1e-12
    if not keep.any():
        return max_d, 0.0
    normals = np.stack([-edges[keep, 1], edges[keep, 0]], axis=1)
    normals /= lengths[keep, None]
    proj = hull @ normals.T
    return max_d, float((proj.max(axis=0) - proj.min(axis=0)).min())


def shape_features(pts_px, mpp):
    """Shape features for one polygon (slide pixels) -> dict in microns."""
    area_px = polygon_area(pts_px)
    perim_px = polygon_perimeter(pts_px)
    hull_px = convex_hull_area(pts_px)
    max_d_px, min_d_px = feret_diameters(pts_px)

    circ = (4.0 * np.pi * area_px / (perim_px ** 2)) if perim_px > 0 else np.nan
    circ = min(circ, 1.0) if np.isfinite(circ) else np.nan

    return {
        "area_um2": area_px * mpp * mpp,
        "length_um": perim_px * mpp,
        "circularity": circ,
        "solidity": (area_px / hull_px) if hull_px > 0 else np.nan,
        "max_diam_um": max_d_px * mpp,
        "min_diam_um": min_d_px * mpp,
    }


def intensity_features_from_patch(rgb_patch, mask=None):
    """Intensity statistics on the deconvolved channels.

    rgb_patch : RGB crop containing the nucleus, at full resolution
    mask      : optional boolean array, same HxW as rgb_patch, True inside
                the nucleus. When given, statistics use only those pixels
                (region=NUCLEUS). When omitted, the whole patch is used.
    """
    hem, dab = deconvolve(rgb_patch)
    out = {}
    for name, chan in (("hem", hem), ("dab", dab)):
        vals = chan[mask] if mask is not None else chan.ravel()
        if vals.size == 0:
            for m in ("mean", "median", "min", "max", "std"):
                out[f"{name}_{m}"] = np.nan
            continue
        out[f"{name}_mean"] = float(vals.mean())
        out[f"{name}_median"] = float(np.median(vals))
        out[f"{name}_min"] = float(vals.min())
        out[f"{name}_max"] = float(vals.max())
        out[f"{name}_std"] = float(vals.std())
    return out


def rasterize_polygon(pts_px, x0, y0, w, h):
    """Boolean mask of the polygon within the given patch window."""
    from matplotlib.path import Path as MplPath
    yy, xx = np.mgrid[y0:y0 + h, x0:x0 + w]
    # Pixel centres, so a pixel counts as inside when its centre is inside.
    pts = np.stack([xx.ravel() + 0.5, yy.ravel() + 0.5], axis=1)
    return MplPath(pts_px).contains_points(pts).reshape(h, w)


def nan_base_features():
    return {k: np.nan for k in BASE_FEATURES}


# ===========================================================================
# Extractor - holds the image source, extracts per nucleus
# ===========================================================================

class NucleusFeatureExtractor:
    """Computes the 16 per-nucleus base features.

    Construct from a WSI path:
        ext = NucleusFeatureExtractor("slide.svs")

    or from an in-memory RGB array (e.g. a tile the segmentation model just
    processed), giving the array's origin in slide coordinates so polygon
    coordinates can stay in slide space:
        ext = NucleusFeatureExtractor(arr, mpp=0.263, origin=(x0, y0))
    """

    def __init__(self, source, mpp=None, origin=(0, 0)):
        self.origin = np.asarray(origin, dtype=float)
        self._slide = None
        self._array = None

        if isinstance(source, np.ndarray):
            if mpp is None:
                raise ValueError("mpp is required when passing an array")
            self._array = source[..., :3]
            self.mpp = float(mpp)
            self.dimensions = (source.shape[1], source.shape[0])
            self.backend = "array"
        else:
            self._slide, self.backend = self._open(source)
            self.mpp = self._read_mpp(self._slide, mpp)
            self.dimensions = self._slide.dimensions


    @staticmethod
    def _open(path):
        try:
            import openslide
            return openslide.OpenSlide(path), "openslide"
        except Exception as e_os:
            try:
                import tiffslide
                return tiffslide.TiffSlide(path), "tiffslide"
            except Exception as e_ts:
                raise RuntimeError(
                    f"Could not open {path}\n  openslide: {e_os}\n"
                    f"  tiffslide: {e_ts}\n"
                    "Install one: pip install openslide-python (or tiffslide)")

    @staticmethod
    def _read_mpp(slide, override):
        if override:
            return float(override)
        for k in ("openslide.mpp-x", "tiffslide.mpp-x", "aperio.MPP"):
            v = slide.properties.get(k)
            if v:
                try:
                    return float(v)
                except ValueError:
                    pass
        raise RuntimeError("Could not read microns-per-pixel from the slide; "
                           "pass mpp explicitly.")

    def read_bbox(self, x0, y0, w, h):
        """Full-resolution RGB crop of the given window, clipped to the
        image. Returns (array, x0, y0) with the actual origin used, or
        None if the window falls entirely outside."""
        W, H = self.dimensions
        if self._array is not None:
            x0 -= int(self.origin[0])
            y0 -= int(self.origin[1])
        x0c, y0c = max(0, x0), max(0, y0)
        x1c, y1c = min(W, x0 + w), min(H, y0 + h)
        if x1c <= x0c or y1c <= y0c:
            return None
        if self._array is not None:
            arr = self._array[y0c:y1c, x0c:x1c]
            return arr, x0c + int(self.origin[0]), y0c + int(self.origin[1])
        arr = np.asarray(self._slide.read_region(
            (x0c, y0c), 0, (x1c - x0c, y1c - y0c)))[..., :3]
        return arr, x0c, y0c

    def extract_one(self, polygon, centroid=None):
        """Base features for a single nucleus.

        polygon : (N,2) array of exterior boundary points, slide pixels
        centroid: optional (x, y); defaults to the polygon's mean point

        Returns a dict with the 16 BASE_FEATURES keys. Intensity is
        measured over pixels inside the polygon. Smoothed features are NOT
        included - they need the surrounding population, so call
        add_smoothed_features() once the whole table is built.
        """
        pts = np.asarray(polygon, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
            return nan_base_features()

        feats = shape_features(pts, self.mpp)

        # Read just the nucleus bounding box, with a 1px margin so boundary
        # pixels aren't clipped by rounding.
        x0 = int(np.floor(pts[:, 0].min())) - 1
        y0 = int(np.floor(pts[:, 1].min())) - 1
        x1 = int(np.ceil(pts[:, 0].max())) + 1
        y1 = int(np.ceil(pts[:, 1].max())) + 1
        got = self.read_bbox(x0, y0, x1 - x0, y1 - y0)
        if got is None:
            feats.update({k: np.nan for k in INTENSITY_FEATURES})
            return feats

        arr, ax0, ay0 = got
        mask = rasterize_polygon(pts, ax0, ay0, arr.shape[1], arr.shape[0])
        if not mask.any():
            # Nucleus smaller than a pixel, or degenerate - fall back to the
            # pixel nearest the centroid so the row isn't lost.
            cx, cy = (pts.mean(axis=0) if centroid is None else centroid)
            ix = int(round(cx)) - ax0
            iy = int(round(cy)) - ay0
            if 0 <= iy < arr.shape[0] and 0 <= ix < arr.shape[1]:
                mask = np.zeros(arr.shape[:2], bool)
                mask[iy, ix] = True
            else:
                feats.update({k: np.nan for k in INTENSITY_FEATURES})
                return feats

        feats.update(intensity_features_from_patch(arr, mask=mask))
        return feats

    def extract_many(self, polygons, centroids=None, progress_every=5000):
        """Loop over polygons, returning (DataFrame of base features,
        (N,2) array of centroids)."""
        rows, cents = [], []
        t0 = time.time()
        for i, poly in enumerate(polygons):
            pts = np.asarray(poly, dtype=np.float64)
            c = (pts.mean(axis=0) if centroids is None
                 else np.asarray(centroids[i], dtype=np.float64))
            rows.append(self.extract_one(pts, centroid=c))
            cents.append(c)
            if progress_every and (i + 1) % progress_every == 0:
                rate = (i + 1) / max(time.time() - t0, 1e-9)
                left = (len(polygons) - i - 1) / rate
                print(f"    {i+1}/{len(polygons)} ({rate:.0f}/s, {left:.0f}s left)")
        return pd.DataFrame(rows, columns=BASE_FEATURES), np.asarray(cents)

    def close(self):
        if self._slide is not None:
            try:
                self._slide.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ===========================================================================
# Smoothing - population level
# ===========================================================================

def smooth_features_single(base_df, centroids_um, fwhm_um, prefix,
                           all_centroids_um=None, all_values=None):
    """Gaussian-weighted neighbourhood average, matching QuPath's
    SmoothFeaturesPlugin(fwhmMicrons=...).

    base_df / centroids_um are the nuclei being described. When
    all_centroids_um is given, neighbours are drawn from that larger set
    instead - so a slide can be smoothed against every detection while
    features are only computed for an annotated subset. all_values holds
    the neighbours' base features (NaN where unknown); neighbours with NaN
    still contribute to the count but not to the averages.
    """
    sigma = fwhm_um / 2.3548200450309493  # FWHM -> sigma
    vals = base_df[BASE_FEATURES].to_numpy(dtype=np.float64)

    # Neighbour COUNTS come from every detection (true tissue density),
    # while weighted AVERAGES use only nuclei that actually have features.
    # Mixing the two would let unmeasured neighbours wash the averages out
    # to NaN.
    counts = np.zeros(len(base_df))
    if all_centroids_um is not None:
        all_tree = cKDTree(all_centroids_um)
        for i, idx in enumerate(all_tree.query_ball_point(centroids_um,
                                                          r=2.0 * fwhm_um)):
            counts[i] = max(len(idx) - 1, 0)

    tree = cKDTree(centroids_um)
    neighbours = tree.query_ball_point(centroids_um, r=2.0 * fwhm_um)

    out = np.full((len(base_df), len(BASE_FEATURES)), np.nan)
    for i, idx in enumerate(neighbours):
        idx = np.asarray(idx, dtype=int)
        if all_centroids_um is None:
            # QuPath's "Nearby detection counts" excludes the object itself.
            counts[i] = max(len(idx) - 1, 0)
        d = np.linalg.norm(centroids_um[idx] - centroids_um[i], axis=1)
        w = np.exp(-(d ** 2) / (2.0 * sigma ** 2))
        block = vals[idx]
        wsum = (w[:, None] * ~np.isnan(block)).sum(axis=0)
        vsum = np.nansum(block * w[:, None], axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[i] = np.where(wsum > 0, vsum / wsum, np.nan)

    res = pd.DataFrame(out, columns=[f"{prefix}_{c}" for c in BASE_FEATURES],
                       index=base_df.index)
    res[f"{prefix}_nearby_count"] = counts
    return res


def add_smoothed_features(base_df, centroids_px, mpp, fwhms=(50.0, 75.0),
                          all_centroids_px=None, all_base_df=None):
    """Append smoothed features to a table of base features.

    base_df          : DataFrame with the 16 BASE_FEATURES columns
    centroids_px     : (N,2) centroids in slide pixels, same order as base_df
    mpp              : microns per pixel
    all_centroids_px : optional (M,2) centroids of EVERY detection on the
                       slide (M >= N). Neighbour counts and weighted
                       averages then reflect true tissue density rather
                       than the density of the annotated subset.
    all_base_df      : features for those M detections; rows may be NaN
                       where features were not computed.
    """
    centroids_um = np.asarray(centroids_px, dtype=np.float64) * mpp
    base = base_df.reset_index(drop=True)

    all_um = all_vals = None
    if all_centroids_px is not None:
        all_um = np.asarray(all_centroids_px, dtype=np.float64) * mpp
        if all_base_df is None:
            # Positions only: neighbour counts become exact, weighted
            # averages fall back to the subset's own values.
            all_vals = np.full((len(all_um), len(BASE_FEATURES)), np.nan)
        else:
            all_vals = all_base_df[BASE_FEATURES].to_numpy(dtype=np.float64)

    parts = [base]
    for f in fwhms:
        prefix = f"sm{int(round(f))}"
        parts.append(smooth_features_single(base, centroids_um, f, prefix,
                                            all_centroids_um=all_um,
                                            all_values=all_vals))
    return pd.concat(parts, axis=1)


# ===========================================================================
# GeoJSON helpers
# ===========================================================================

def load_geojson(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        gj = json.load(f)
    if isinstance(gj, dict) and gj.get("type") == "FeatureCollection":
        return gj["features"]
    if isinstance(gj, list):
        return gj
    if isinstance(gj, dict) and gj.get("type") == "Feature":
        return [gj]
    raise ValueError(f"Unrecognised GeoJSON structure in {path}")


def feature_polygons(feat):
    """Yields exterior rings as (N,2) arrays. Holes are ignored - nuclei
    are simple shapes and shape measurements use the outer boundary."""
    geom = feat.get("geometry") or {}
    coords = geom.get("coordinates")
    if coords is None:
        return
    if geom.get("type") == "Polygon":
        yield np.asarray(coords[0], dtype=np.float64)
    elif geom.get("type") == "MultiPolygon":
        for poly in coords:
            yield np.asarray(poly[0], dtype=np.float64)


def polygons_from_geojson(path, min_points=3):
    out = []
    for feat in load_geojson(path):
        for ring in feature_polygons(feat):
            if len(ring) >= min_points:
                out.append(ring)
    return out


def assign_labels(centroids_px, annotation_path, verbose=True):
    """Point-in-polygon against ground-truth regions, vectorised.

    Where a nucleus falls in nested regions the smallest (most specific)
    wins, matching assign_cells_from_annotations.groovy.

    Regions are tested smallest-area first, so the first hit for a nucleus
    is already its most specific region and it can be dropped from further
    testing. Combined with a batched bounding-box prefilter this turns
    O(nuclei x regions) Python-level work into a handful of numpy passes.
    """
    from matplotlib.path import Path as MplPath

    centroids_px = np.asarray(centroids_px, dtype=np.float64)
    n = len(centroids_px)

    regions = []
    for f in load_geojson(annotation_path):
        props = f.get("properties", {}) or {}
        cls = props.get("classification")
        nm = cls.get("name") if isinstance(cls, dict) else props.get("name")
        if not nm:
            continue
        for ring in feature_polygons(f):
            if len(ring) < 3:
                continue
            regions.append((nm, ring, polygon_area(ring),
                            ring[:, 0].min(), ring[:, 0].max(),
                            ring[:, 1].min(), ring[:, 1].max()))
    if not regions:
        if verbose:
            print("    no classified annotation regions found")
        return [None] * n

    regions.sort(key=lambda r: r[2])  # smallest area first
    if verbose:
        print(f"    {len(regions)} annotation regions, testing {n} nuclei")

    labels = np.full(n, None, dtype=object)
    unassigned = np.arange(n)
    cx, cy = centroids_px[:, 0], centroids_px[:, 1]

    for nm, ring, _area, x0, x1, y0, y1 in regions:
        if unassigned.size == 0:
            break
        # Cheap vectorised bbox reject over everything still unassigned.
        sub = unassigned
        m = ((cx[sub] >= x0) & (cx[sub] <= x1) &
             (cy[sub] >= y0) & (cy[sub] <= y1))
        cand = sub[m]
        if cand.size == 0:
            continue
        inside = MplPath(ring).contains_points(centroids_px[cand])
        hit = cand[inside]
        if hit.size:
            labels[hit] = nm
            unassigned = unassigned[~np.isin(unassigned, hit, assume_unique=True)]

    return labels.tolist()


# ===========================================================================
# CLI
# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi", default=DEFAULT_WSI)
    ap.add_argument("--geojson", default=DEFAULT_GEOJSON,
                    help="Detection polygons (.geojson or .geojson.gz)")
    ap.add_argument("--annotations", default=DEFAULT_ANNOTATIONS,
                    help="GT region annotations, adds a 'label' column")
    ap.add_argument("--no_annotations", action="store_true",
                    help="Skip label assignment even if a default is set")
    ap.add_argument("--keep_unlabeled", action="store_true",
                    help="Extract features for ALL detections, not just those "
                         "inside annotations. Needed for inference on a whole "
                         "slide; wasteful when building a training set.")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--mpp", type=float, default=None)
    ap.add_argument("--min_area_um2", type=float, default=0.0)
    ap.add_argument("--max_area_um2", type=float, default=float("inf"))
    args = ap.parse_args()

    annotations = None if args.no_annotations else args.annotations

    for label, path in (("WSI", args.wsi), ("detections", args.geojson)):
        if not os.path.exists(path):
            raise SystemExit(f"{label} not found: {path}\n"
                             f"Edit DEFAULT_* at the top of this script, or pass "
                             f"--wsi / --geojson explicitly.")
    if annotations and not os.path.exists(annotations):
        print(f"  WARNING: annotations not found ({annotations}) - "
              f"continuing without labels")
        annotations = None

    name = os.path.splitext(os.path.basename(args.wsi))[0]
    print(f"=== {name} ===")
    print(f"  wsi        : {args.wsi}")
    print(f"  detections : {args.geojson}")
    print(f"  annotations: {annotations or '(none)'}")
    print(f"  out        : {args.out}")

    print("  loading detections...")
    t0 = time.time()
    polys = polygons_from_geojson(args.geojson)
    print(f"  {len(polys)} nuclei ({time.time()-t0:.1f}s)")
    if not polys:
        raise SystemExit("No usable polygons found.")

    t0 = time.time()
    centroids_all = np.array([p.mean(axis=0) for p in polys])
    print(f"  centroids ({time.time()-t0:.1f}s)")
    # Keep every detection's position: features are expensive (one image
    # read each) so they are computed only for annotated nuclei, but
    # neighbour counts should still reflect true tissue density.
    centroids_every_detection = centroids_all.copy()

    labels = None
    if annotations:
        print("  assigning labels...")
        t0 = time.time()
        labels = assign_labels(centroids_all, annotations)
        n_lab = sum(l is not None for l in labels)
        print(f"    {n_lab}/{len(labels)} nuclei inside an annotation "
              f"({time.time()-t0:.1f}s)")
        for k, v in pd.Series([l for l in labels if l]).value_counts().items():
            print(f"      {k}: {v}")

        if not args.keep_unlabeled:
            keep = [i for i, l in enumerate(labels) if l is not None]
            if not keep:
                raise SystemExit("No nuclei inside any annotation - nothing to do.")
            print(f"  extracting features for {len(keep)} labelled nuclei "
                  f"(skipping {len(polys) - len(keep)} outside annotations)")
            polys = [polys[i] for i in keep]
            centroids_all = centroids_all[keep]
            labels = [labels[i] for i in keep]

    with NucleusFeatureExtractor(args.wsi, mpp=args.mpp) as ext:
        print(f"  {ext.backend}, {ext.dimensions[0]}x{ext.dimensions[1]} px, "
              f"{ext.mpp:.6f} um/px")
        print("  base features (shape + intensity)...")
        base, centroids = ext.extract_many(polys, centroids=centroids_all)
        mpp = ext.mpp

    if args.min_area_um2 > 0 or np.isfinite(args.max_area_um2):
        keep = ((base["area_um2"] >= args.min_area_um2) &
                (base["area_um2"] <= args.max_area_um2)).to_numpy()
        n_drop = int((~keep).sum())
        if n_drop:
            print(f"    {n_drop} dropped by size filter")
            base = base[keep].reset_index(drop=True)
            centroids = centroids[keep]
            if labels is not None:
                labels = [l for l, k in zip(labels, keep) if k]

    print("  smoothing (50um, 75um)...")
    # Neighbour counts use every detection on the slide, matching QuPath,
    # even though features were only extracted for the annotated subset.
    df = add_smoothed_features(base, centroids, mpp,
                               all_centroids_px=centroids_every_detection)

    df.insert(0, "cy_wsi", centroids[:, 1])
    df.insert(0, "cx_wsi", centroids[:, 0])
    df.insert(0, "nucleus_id", range(len(df)))
    df.insert(0, "wsi_name", name)

    if labels is not None:
        df["label"] = labels
        df["is_ground_truth"] = [l is not None for l in labels]

    df.to_csv(args.out, index=False)
    print(f"\nWrote {len(df)} nuclei x {len(BASE_FEATURES)*3 + 2} features "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
