import qupath.lib.common.GeneralTools

// ============================================================
// Exports LABELED annotations to GeoJSON only (full polygon geometry) -
// Run via: Automate > Run for project
// ============================================================

// ---------------- SETTINGS ----------------
// Set to null/empty to export ALL annotations regardless of class.
def keepClasses = [
    "Tumor", "Stroma", "Immune cells",
    "Normal-GI-Mucosa", "Normal-foveola", "Normal-Squamous", "Normal-Brunner",
    "Normal-endometrial-Ep", "Normal-Glands", "Normal-Smooth-Muscle",
    "Necrosis", "Other",
    "Ghost",  // exported so it correctly "wins" if nested inside a larger
              // Tumor/Stroma annotation (smallest-region-wins in
              // assign_labels) - dropped later by Python's collapse_label,
              // not excluded here. Excluding it HERE would let a nucleus
              // meant to be carved out instead inherit the surrounding
              // region's real class.
    // TEXTURE annotations (real ground truth - collapsed to Tumor/Stroma/
    // Immune in Python) and MIXED-* (NOT ground truth - dropped in Python,
    // same reasoning as Ghost above: must be exported here so nesting
    // resolves correctly, exclusion happens downstream).
    "TextTumHigh", "TextTumLow", "TextStromHigh", "TextStromLow", "TextImmune",
    "MIXED-MostStom", "MIXED-MostTum"
] as Set

boolean EXPORT_ALL_CLASSES = true   // true = ignore keepClasses, export everything

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
if (annotations.isEmpty()) { print "No matching (labeled) annotations - skipping."; return }
print "Annotations to export: " + annotations.size()

def classCounts = [:].withDefault { 0 }
annotations.each { classCounts[it.getPathClass()?.getName() ?: "(unclassified)"]++ }
classCounts.sort { -it.value }.each { k, v -> print "  " + k + ": " + v }

def outDir = buildFilePath(PROJECT_BASE_DIR, 'annotations-labeled', name)
mkdirs(outDir)

// ---------------- GEOJSON (full geometry) ----------------
def geojsonPath = buildFilePath(outDir, name + "_annotations.geojson")
exportObjectsToGeoJson(annotations, geojsonPath, "FEATURE_COLLECTION")
print "GeoJSON: " + geojsonPath

print "Done - " + annotations.size() + " annotations exported."
