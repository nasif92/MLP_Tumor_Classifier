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
    if key in ("non-tumor", "nontumor"):
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
