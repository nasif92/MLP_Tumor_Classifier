import pandas as pd

phase1="/mnt/NAS/MAGEE/October-DataSet-QP/phase1_annotations"
phase2="/mnt/NAS/QuPath_Projects_AA/cellpose-dino-cls/qmlp-bcw_oct_0_9/phase2_annotations"
ref_set = "/mnt/NAS/QuPath_Projects_AA/cellpose-dino-cls/qp-6_reference_slides-no_artifacts/annotations_and_detections"
ref_set_no_cells ="/mnt/NAS/QuPath_Projects_AA/cellpose-dino-cls/qp-6_reference_slides-no_artifacts-no_cells/annotations_and_detections"
all_features = "/mnt/NAS/BreastCancerWSIs-Detections/OCTOBER-2024/cellpose-dino/seg"

## TODO: New detection binary file and annotated detections file
ref_set_er = "/mnt/NAS/QuPath_Projects_AA/cellpose-dino-cls/qp-6_reference_slides-no_artifacts-no_cells-er/annotations_and_detections-labeled-binary"
ref_set_er_all_detections = "/mnt/NAS/QuPath_Projects_AA/cellpose-dino-cls/qp-6_reference_slides-no_artifacts-no_cells-er/detections-binary"
ref_set_ki67 = "/mnt/NAS/QuPath_Projects_AA/cellpose-dino-cls/qp-6_reference_slides-no_artifacts-no_cells-ki67/annotations_and_detections-labeled-binary"
ref_set_ki67_all_detections = "/mnt/NAS/QuPath_Projects_AA/cellpose-dino-cls/qp-6_reference_slides-no_artifacts-no_cells-ki67/detections-binary"

#TODO: setting annotations for ER here for now
ref_set_ann_pr = "/mnt/NAS/QuPath_Projects_AA/cellpose-dino-cls/qp-6_reference_slides-no_artifacts-no_cells/annotations-labeled"
ref_set_ann_er ="/mnt/NAS/QuPath_Projects_AA/cellpose-dino-cls/qp-6_reference_slides-no_artifacts-no_cells-er/annotations-labeled"
ref_set_ann_ki67 = "/mnt/NAS/QuPath_Projects_AA/cellpose-dino-cls/qp-6_reference_slides-no_artifacts-no_cells-ki67/annotations-labeled"
wsi_path = "/mnt/NAS/Abhineet/BreastCancerWSIs/OCTOBER-2024"
# Single source of truth for the subtype vocabulary, shared by train_deploy.py
# (which trains on it) and extract_features.py's --batch mode (which needs it
# to recognize a subtype token, e.g. "ER", inside a slide name).
DEFAULT_SUBTYPES = ["ER", "PR", "Ki67"]

def collapse_label(raw):
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    raw = str(raw).strip()
    if raw == "" or raw.lower().startswith("ignore"):
        return None

    key = raw.lower()

    if key == "tumor":
        return "Tumor"
    if key in ("texttumhigh", "texttumlow"):
        return "Tumor"

    if key in ("stroma", "immune cells", "necrosis", "other",
               "textstromhigh", "textstromlow", "textimmune"):
        return "Non Tumor"
    if key in ("non-tumor", "nontumor", "non tumor"):
        return "Non Tumor"

    if key.startswith("normal"):
        return "Normal"

    # Deliberately excluded from training: artifacts, QC flags, ambiguous
    # regions - never given a real class, regardless of naming variant.
    if key == "ghost":
        return None
    if key in ("bad tissue", "badtissue"):
        return None

    return None  # anything unrecognized is dropped, not silently misclassified
    # if raw == "MIXED-MostStom":
    #     return "Stroma"
    # if raw == "MIXED-MostTum":
    #     return "Tumor"
    # "Ghost" (and anything else not explicitly mapped above) falls through
    # here and is dropped - Gilbert marks these as a class deliberately
    # excluded from the classifier, not a real tissue category.

    # --------------------------------------------------------------------------
# Feature sets for train_deploy.py's MLP. "default" excludes every
# "Smoothed: ..." measurement (neighborhood/context features, including
# "Nearby detection counts" - every occurrence of it is itself a Smoothed:
# variant). diag_ki67 flagged these as both the strongest train-discriminative
# features AND the ones that shift most between train and reference slides -
# consistent with the model learning a shortcut from how annotation regions
# are drawn (homogeneous by construction) rather than real biological signal.
# "all" is every measurement seen on a real detection, kept for reference and
# for reverting to the un-filtered feature set. Edit these lists directly, or
# add a new named set, to control exactly what the MLP is allowed to see -
# train_deploy.py just intersects a chosen set with whatever's in the data.
FEATURE_SETS = {
    "all": [
    'Area µm^2',
    'Circularity',
    'Custom: ROI Entropy (DAB)',
    'Custom: ROI Entropy (Hematoxylin)',
    'Custom: ROI Entropy (OD Sum)',
    'Length µm',
    'Max diameter µm',
    'Min diameter µm',
    'ROI: 0.50 µm per pixel: DAB: Haralick Angular second moment (F0)',
    'ROI: 0.50 µm per pixel: DAB: Haralick Contrast (F1)',
    'ROI: 0.50 µm per pixel: DAB: Haralick Correlation (F2)',
    'ROI: 0.50 µm per pixel: DAB: Haralick Difference entropy (F10)',
    'ROI: 0.50 µm per pixel: DAB: Haralick Difference variance (F9)',
    'ROI: 0.50 µm per pixel: DAB: Haralick Entropy (F8)',
    'ROI: 0.50 µm per pixel: DAB: Haralick Information measure of correlation 1 (F11)',
    'ROI: 0.50 µm per pixel: DAB: Haralick Information measure of correlation 2 (F12)',
    'ROI: 0.50 µm per pixel: DAB: Haralick Inverse difference moment (F4)',
    'ROI: 0.50 µm per pixel: DAB: Haralick Sum average (F5)',
    'ROI: 0.50 µm per pixel: DAB: Haralick Sum entropy (F7)',
    'ROI: 0.50 µm per pixel: DAB: Haralick Sum of squares (F3)',
    'ROI: 0.50 µm per pixel: DAB: Haralick Sum variance (F6)',
    'ROI: 0.50 µm per pixel: DAB: Max',
    'ROI: 0.50 µm per pixel: DAB: Mean',
    'ROI: 0.50 µm per pixel: DAB: Median',
    'ROI: 0.50 µm per pixel: DAB: Min',
    'ROI: 0.50 µm per pixel: DAB: Std.dev.',
    'ROI: 0.50 µm per pixel: Hematoxylin: Haralick Angular second moment (F0)',
    'ROI: 0.50 µm per pixel: Hematoxylin: Haralick Contrast (F1)',
    'ROI: 0.50 µm per pixel: Hematoxylin: Haralick Correlation (F2)',
    'ROI: 0.50 µm per pixel: Hematoxylin: Haralick Difference entropy (F10)',
    'ROI: 0.50 µm per pixel: Hematoxylin: Haralick Difference variance (F9)',
    'ROI: 0.50 µm per pixel: Hematoxylin: Haralick Entropy (F8)',
    'ROI: 0.50 µm per pixel: Hematoxylin: Haralick Information measure of correlation 1 (F11)',
    'ROI: 0.50 µm per pixel: Hematoxylin: Haralick Information measure of correlation 2 (F12)',
    'ROI: 0.50 µm per pixel: Hematoxylin: Haralick Inverse difference moment (F4)',
    'ROI: 0.50 µm per pixel: Hematoxylin: Haralick Sum average (F5)',
    'ROI: 0.50 µm per pixel: Hematoxylin: Haralick Sum entropy (F7)',
    'ROI: 0.50 µm per pixel: Hematoxylin: Haralick Sum of squares (F3)',
    'ROI: 0.50 µm per pixel: Hematoxylin: Haralick Sum variance (F6)',
    'ROI: 0.50 µm per pixel: Hematoxylin: Max',
    'ROI: 0.50 µm per pixel: Hematoxylin: Mean',
    'ROI: 0.50 µm per pixel: Hematoxylin: Median',
    'ROI: 0.50 µm per pixel: Hematoxylin: Min',
    'ROI: 0.50 µm per pixel: Hematoxylin: Std.dev.',
    'ROI: 0.50 µm per pixel: OD Sum: Haralick Angular second moment (F0)',
    'ROI: 0.50 µm per pixel: OD Sum: Haralick Contrast (F1)',
    'ROI: 0.50 µm per pixel: OD Sum: Haralick Correlation (F2)',
    'ROI: 0.50 µm per pixel: OD Sum: Haralick Difference entropy (F10)',
    'ROI: 0.50 µm per pixel: OD Sum: Haralick Difference variance (F9)',
    'ROI: 0.50 µm per pixel: OD Sum: Haralick Entropy (F8)',
    'ROI: 0.50 µm per pixel: OD Sum: Haralick Information measure of correlation 1 (F11)',
    'ROI: 0.50 µm per pixel: OD Sum: Haralick Information measure of correlation 2 (F12)',
    'ROI: 0.50 µm per pixel: OD Sum: Haralick Inverse difference moment (F4)',
    'ROI: 0.50 µm per pixel: OD Sum: Haralick Sum average (F5)',
    'ROI: 0.50 µm per pixel: OD Sum: Haralick Sum entropy (F7)',
    'ROI: 0.50 µm per pixel: OD Sum: Haralick Sum of squares (F3)',
    'ROI: 0.50 µm per pixel: OD Sum: Haralick Sum variance (F6)',
    'ROI: 0.50 µm per pixel: OD Sum: Max',
    'ROI: 0.50 µm per pixel: OD Sum: Mean',
    'ROI: 0.50 µm per pixel: OD Sum: Median',
    'ROI: 0.50 µm per pixel: OD Sum: Min',
    'ROI: 0.50 µm per pixel: OD Sum: Std.dev.',
    'Smoothed: 50 µm: Area µm^2',
    'Smoothed: 50 µm: Circularity',
    'Smoothed: 50 µm: Length µm',
    'Smoothed: 50 µm: Max diameter µm',
    'Smoothed: 50 µm: Min diameter µm',
    'Smoothed: 50 µm: Nearby detection counts',
    'Smoothed: 50 µm: ROI: 0.50 µm per pixel: DAB: Max',
    'Smoothed: 50 µm: ROI: 0.50 µm per pixel: DAB: Mean',
    'Smoothed: 50 µm: ROI: 0.50 µm per pixel: DAB: Median',
    'Smoothed: 50 µm: ROI: 0.50 µm per pixel: DAB: Min',
    'Smoothed: 50 µm: ROI: 0.50 µm per pixel: DAB: Std.dev.',
    'Smoothed: 50 µm: ROI: 0.50 µm per pixel: Hematoxylin: Max',
    'Smoothed: 50 µm: ROI: 0.50 µm per pixel: Hematoxylin: Mean',
    'Smoothed: 50 µm: ROI: 0.50 µm per pixel: Hematoxylin: Median',
    'Smoothed: 50 µm: ROI: 0.50 µm per pixel: Hematoxylin: Min',
    'Smoothed: 50 µm: ROI: 0.50 µm per pixel: Hematoxylin: Std.dev.',
    'Smoothed: 50 µm: ROI: 0.50 µm per pixel: OD Sum: Max',
    'Smoothed: 50 µm: ROI: 0.50 µm per pixel: OD Sum: Mean',
    'Smoothed: 50 µm: ROI: 0.50 µm per pixel: OD Sum: Median',
    'Smoothed: 50 µm: ROI: 0.50 µm per pixel: OD Sum: Min',
    'Smoothed: 50 µm: ROI: 0.50 µm per pixel: OD Sum: Std.dev.',
    'Smoothed: 50 µm: Solidity',
    'Smoothed: 75 µm: Area µm^2',
    'Smoothed: 75 µm: Circularity',
    'Smoothed: 75 µm: Length µm',
    'Smoothed: 75 µm: Max diameter µm',
    'Smoothed: 75 µm: Min diameter µm',
    'Smoothed: 75 µm: Nearby detection counts',
    'Smoothed: 75 µm: ROI: 0.50 µm per pixel: DAB: Max',
    'Smoothed: 75 µm: ROI: 0.50 µm per pixel: DAB: Mean',
    'Smoothed: 75 µm: ROI: 0.50 µm per pixel: DAB: Median',
    'Smoothed: 75 µm: ROI: 0.50 µm per pixel: DAB: Min',
    'Smoothed: 75 µm: ROI: 0.50 µm per pixel: DAB: Std.dev.',
    'Smoothed: 75 µm: ROI: 0.50 µm per pixel: Hematoxylin: Max',
    'Smoothed: 75 µm: ROI: 0.50 µm per pixel: Hematoxylin: Mean',
    'Smoothed: 75 µm: ROI: 0.50 µm per pixel: Hematoxylin: Median',
    'Smoothed: 75 µm: ROI: 0.50 µm per pixel: Hematoxylin: Min',
    'Smoothed: 75 µm: ROI: 0.50 µm per pixel: Hematoxylin: Std.dev.',
    'Smoothed: 75 µm: ROI: 0.50 µm per pixel: OD Sum: Max',
    'Smoothed: 75 µm: ROI: 0.50 µm per pixel: OD Sum: Mean',
    'Smoothed: 75 µm: ROI: 0.50 µm per pixel: OD Sum: Median',
    'Smoothed: 75 µm: ROI: 0.50 µm per pixel: OD Sum: Min',
    'Smoothed: 75 µm: ROI: 0.50 µm per pixel: OD Sum: Std.dev.',
    'Smoothed: 75 µm: Solidity',
    'Solidity',
],
    "default": [
    'Area µm^2',
    'Circularity',
    'Custom: ROI Entropy (DAB)',
    'Custom: ROI Entropy (Hematoxylin)',
    'Custom: ROI Entropy (OD Sum)',
    'Length µm',
    'Max diameter µm',
    'Min diameter µm',
    'ROI: 0.50 µm per pixel: DAB: Haralick Angular second moment (F0)',
    'ROI: 0.50 µm per pixel: DAB: Haralick Contrast (F1)',
    'ROI: 0.50 µm per pixel: DAB: Haralick Correlation (F2)',
    'ROI: 0.50 µm per pixel: DAB: Haralick Difference entropy (F10)',
    'ROI: 0.50 µm per pixel: DAB: Haralick Difference variance (F9)',
    'ROI: 0.50 µm per pixel: DAB: Haralick Entropy (F8)',
    'ROI: 0.50 µm per pixel: DAB: Haralick Information measure of correlation 1 (F11)',
    'ROI: 0.50 µm per pixel: DAB: Haralick Information measure of correlation 2 (F12)',
    'ROI: 0.50 µm per pixel: DAB: Haralick Inverse difference moment (F4)',
    'ROI: 0.50 µm per pixel: DAB: Haralick Sum average (F5)',
    'ROI: 0.50 µm per pixel: DAB: Haralick Sum entropy (F7)',
    'ROI: 0.50 µm per pixel: DAB: Haralick Sum of squares (F3)',
    'ROI: 0.50 µm per pixel: DAB: Haralick Sum variance (F6)',
    'ROI: 0.50 µm per pixel: DAB: Max',
    'ROI: 0.50 µm per pixel: DAB: Mean',
    'ROI: 0.50 µm per pixel: DAB: Median',
    'ROI: 0.50 µm per pixel: DAB: Min',
    'ROI: 0.50 µm per pixel: DAB: Std.dev.',
    'ROI: 0.50 µm per pixel: Hematoxylin: Haralick Angular second moment (F0)',
    'ROI: 0.50 µm per pixel: Hematoxylin: Haralick Contrast (F1)',
    'ROI: 0.50 µm per pixel: Hematoxylin: Haralick Correlation (F2)',
    'ROI: 0.50 µm per pixel: Hematoxylin: Haralick Difference entropy (F10)',
    'ROI: 0.50 µm per pixel: Hematoxylin: Haralick Difference variance (F9)',
    'ROI: 0.50 µm per pixel: Hematoxylin: Haralick Entropy (F8)',
    'ROI: 0.50 µm per pixel: Hematoxylin: Haralick Information measure of correlation 1 (F11)',
    'ROI: 0.50 µm per pixel: Hematoxylin: Haralick Information measure of correlation 2 (F12)',
    'ROI: 0.50 µm per pixel: Hematoxylin: Haralick Inverse difference moment (F4)',
    'ROI: 0.50 µm per pixel: Hematoxylin: Haralick Sum average (F5)',
    'ROI: 0.50 µm per pixel: Hematoxylin: Haralick Sum entropy (F7)',
    'ROI: 0.50 µm per pixel: Hematoxylin: Haralick Sum of squares (F3)',
    'ROI: 0.50 µm per pixel: Hematoxylin: Haralick Sum variance (F6)',
    'ROI: 0.50 µm per pixel: Hematoxylin: Max',
    'ROI: 0.50 µm per pixel: Hematoxylin: Mean',
    'ROI: 0.50 µm per pixel: Hematoxylin: Median',
    'ROI: 0.50 µm per pixel: Hematoxylin: Min',
    'ROI: 0.50 µm per pixel: Hematoxylin: Std.dev.',
    'ROI: 0.50 µm per pixel: OD Sum: Haralick Angular second moment (F0)',
    'ROI: 0.50 µm per pixel: OD Sum: Haralick Contrast (F1)',
    'ROI: 0.50 µm per pixel: OD Sum: Haralick Correlation (F2)',
    'ROI: 0.50 µm per pixel: OD Sum: Haralick Difference entropy (F10)',
    'ROI: 0.50 µm per pixel: OD Sum: Haralick Difference variance (F9)',
    'ROI: 0.50 µm per pixel: OD Sum: Haralick Entropy (F8)',
    'ROI: 0.50 µm per pixel: OD Sum: Haralick Information measure of correlation 1 (F11)',
    'ROI: 0.50 µm per pixel: OD Sum: Haralick Information measure of correlation 2 (F12)',
    'ROI: 0.50 µm per pixel: OD Sum: Haralick Inverse difference moment (F4)',
    'ROI: 0.50 µm per pixel: OD Sum: Haralick Sum average (F5)',
    'ROI: 0.50 µm per pixel: OD Sum: Haralick Sum entropy (F7)',
    'ROI: 0.50 µm per pixel: OD Sum: Haralick Sum of squares (F3)',
    'ROI: 0.50 µm per pixel: OD Sum: Haralick Sum variance (F6)',
    'ROI: 0.50 µm per pixel: OD Sum: Max',
    'ROI: 0.50 µm per pixel: OD Sum: Mean',
    'ROI: 0.50 µm per pixel: OD Sum: Median',
    'ROI: 0.50 µm per pixel: OD Sum: Min',
    'ROI: 0.50 µm per pixel: OD Sum: Std.dev.',
    'Solidity',
],
}
