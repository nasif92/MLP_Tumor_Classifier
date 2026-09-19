import pandas as pd

phase1="/mnt/NAS/MAGEE/October-DataSet-QP/phase1_annotations"
phase2="/mnt/NAS/QuPath_Projects_AA/cellpose-dino-cls/qmlp-bcw_oct_0_9/phase2_annotations"
ref_set = "/mnt/NAS/QuPath_Projects_AA/cellpose-dino-cls/qp-6_reference_slides-no_artifacts/annotations_and_detections"
ref_set_no_cells ="/mnt/NAS/QuPath_Projects_AA/cellpose-dino-cls/qp-6_reference_slides-no_artifacts-no_cells/annotations_and_detections"
all_features = "/mnt/NAS/BreastCancerWSIs-Detections/OCTOBER-2024/cellpose-dino/seg"

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
    if raw == "Tumor":
        return "Tumor"
    if raw == "Stroma":
        return "Non Tumor"
    if raw == "Immune cells":
        return "Non Tumor"
    if raw.startswith("Normal"):
        return "Normal"
    if raw in ("Necrosis", "Other"):
        return "Non Tumor"
    if raw in ("TextTumHigh", "TextTumLow"):
        return "Tumor"
    if raw in ("TextStromHigh", "TextStromLow"):
        return "Non Tumor"
    if raw == "TextImmune":
        return "Non Tumor"
    if raw == "Ghost":
        return None
    if raw == "Other":
        return None

    # if raw == "MIXED-MostStom":
    #     return "Stroma"
    # if raw == "MIXED-MostTum":
    #     return "Tumor"
    # "Ghost" (and anything else not explicitly mapped above) falls through
    # here and is dropped - Gilbert marks these as a class deliberately
    # excluded from the classifier, not a real tissue category.
    return None