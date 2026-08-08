import qupath.lib.common.GeneralTools

// ============================================================
// Exports annotations to GeoJSON (full polygon geometry) plus a
// summary CSV (one row per annotation: class, area, bounding box).
//
// Run via: Automate > Run for project
// ============================================================

// ---------------- SETTINGS ----------------
// Set to null/empty to export ALL annotations regardless of class.
def keepClasses = [
    "Tumor", "Stroma", "Immune cells",
    "Normal-GI-Mucosa", "Normal-foveola", "Normal-Squamous", "Normal-Brunner",
    "Normal-endometrial-Ep", "Normal-Glands", "Normal-Smooth-Muscle",
    "Necrosis", "Other"
] as Set

boolean EXPORT_ALL_CLASSES = false   // true = ignore keepClasses, export everything

def imageData = getCurrentImageData()
if (imageData == null) { print "No image open!"; return }
def hierarchy = imageData.getHierarchy()
def server = imageData.getServer()
def name = GeneralTools.getNameWithoutExtension(server.getMetadata().getName())
print "=== " + name + " ==="

// ---------------- COLLECT ANNOTATIONS ----------------
def allAnnotations = hierarchy.getAnnotationObjects()
print "Total annotations on slide: " + allAnnotations.size()

def annotations = EXPORT_ALL_CLASSES ? allAnnotations : allAnnotations.findAll { ann ->
    def nm = ann.getPathClass()?.getName()?.trim()
    nm != null && keepClasses.contains(nm)
}
if (annotations.isEmpty()) { print "No matching annotations - skipping."; return }
print "Annotations to export: " + annotations.size()

def classCounts = [:].withDefault { 0 }
annotations.each { classCounts[it.getPathClass()?.getName() ?: "(unclassified)"]++ }
classCounts.sort { -it.value }.each { k, v -> print "  " + k + ": " + v }

// Stable ID per annotation, written into the measurement list so it carries
// through to the GeoJSON properties and can be joined against the CSV.
annotations.eachWithIndex { ann, i ->
    ann.getMeasurementList().put("__export_annotation_id", i as double)
}

def outDir = buildFilePath(PROJECT_BASE_DIR, 'annotation_exports', name)
mkdirs(outDir)

// ---------------- GEOJSON (full geometry) ----------------
def geojsonPath = buildFilePath(outDir, name + "_annotations.geojson")
exportObjectsToGeoJson(annotations, geojsonPath, "FEATURE_COLLECTION")
print "GeoJSON: " + geojsonPath

// ---------------- SUMMARY CSV ----------------
def cal = server.getPixelCalibration()
double pxArea = cal.getPixelWidthMicrons() * cal.getPixelHeightMicrons()   // um^2 per pixel
boolean hasMicrons = !Double.isNaN(pxArea) && pxArea > 0

def csvFile = new File(buildFilePath(outDir, name + "_annotations.csv"))
csvFile.withWriter { writer ->
    writer.writeLine(["wsi_name","annotation_id","class","name",
                      "area_px2","area_um2","perimeter_px",
                      "bbox_x","bbox_y","bbox_w","bbox_h",
                      "centroid_x","centroid_y",
                      "roi_type","n_points","n_child_objects"].join(","))
    annotations.eachWithIndex { ann, i ->
        def roi = ann.getROI()
        double areaPx = roi.getArea()
        def clsName = ann.getPathClass()?.getName() ?: ""
        // Annotation display name can contain commas - quote it.
        def annName = (ann.getName() ?: "").replace("\"", "'")
        int nPoints = -1
        try { nPoints = roi.getAllPoints().size() } catch (Exception e) { }
        writer.writeLine([
            name, i, clsName, "\"" + annName + "\"",
            String.format("%.2f", areaPx),
            hasMicrons ? String.format("%.2f", areaPx * pxArea) : "",
            String.format("%.2f", roi.getLength()),
            String.format("%.2f", roi.getBoundsX()), String.format("%.2f", roi.getBoundsY()),
            String.format("%.2f", roi.getBoundsWidth()), String.format("%.2f", roi.getBoundsHeight()),
            String.format("%.2f", roi.getCentroidX()), String.format("%.2f", roi.getCentroidY()),
            roi.getRoiName(), nPoints,
            ann.nChildObjects()
        ].join(","))
    }
}
print "CSV: " + csvFile.getPath()
print "Done - " + annotations.size() + " annotations exported."
