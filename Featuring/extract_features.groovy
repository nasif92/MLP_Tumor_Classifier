import qupath.lib.objects.PathObjects
import qupath.lib.objects.PathCellObject
import qupath.lib.common.GeneralTools
import qupath.lib.gui.prefs.PathPrefs

// ============================================================
// Computes the 50 reference-classifier features on Cellpose-DINO
// detections.
//
// Plain PathDetectionObjects have no nucleus ROI, so
// IntensityFeaturesPlugin(region=NUCLEUS) silently produces nothing -
// which is why only "Square: ..." names ever appeared. StarDist made
// cell objects, so it worked there.
//
// This converts each detection into a cell object using its own polygon
// as BOTH the cell boundary and the nucleus, so region=NUCLEUS measures
// inside the nucleus outline. That matches the Python extractor.
//
// The measurement names produced are printed at the end - check them
// against the classifier's expected list before trusting the output.
// ============================================================

def imageData = getCurrentImageData()
if (imageData == null) { print "No image open!"; return }
def hierarchy = imageData.getHierarchy()
def server = imageData.getServer()
def name = GeneralTools.getNameWithoutExtension(server.getMetadata().getName())
print "=== " + name + " ==="

// Native resolution, so intensities are measured on real pixels rather
// than a coarse downsample (a ~7um nucleus is only ~3px across at 2um/px).
double mpp = server.getPixelCalibration().getAveragedPixelSizeMicrons()
if (Double.isNaN(mpp)) { print "No pixel calibration on this image."; return }
print String.format("Pixel size: %.6f um", mpp)

setImageType('BRIGHTFIELD_H_DAB')
setColorDeconvolutionStains('{"Name" : "H-DAB default", "Stain 1" : "Hematoxylin", "Values 1" : "0.6511078257574492 0.7011930431234068 0.29049426072255424", "Stain 2" : "DAB", "Values 2" : "0.2691668720495607 0.5682411743268503 0.7775931859209531", "Background" : " 255 255 255"}')

// ---------------- 1. CONVERT DETECTIONS TO CELL OBJECTS ----------------
def existing = getDetectionObjects()
if (existing.isEmpty()) { print "No detections."; return }
print "Detections: " + existing.size()

int alreadyCells = existing.count { it instanceof PathCellObject }
if (alreadyCells == existing.size()) {
    print "Already cell objects - skipping conversion."
} else {
    print "Converting " + existing.size() + " detections to cell objects..."
    def cells = existing.collect { det ->
        def roi = det.getROI()
        // Same ROI for cell and nucleus: region=NUCLEUS then measures
        // inside the segmented nucleus outline.
        // Old measurements are deliberately NOT carried over - every
        // feature is recomputed below, and copying between measurement
        // lists during iteration throws ConcurrentModificationException.
        return PathObjects.createCellObject(roi, roi, det.getPathClass(), null)
    }
    // Replace in one hierarchy operation - removing and adding one at a
    // time on ~200k objects is extremely slow.
    hierarchy.removeObjects(existing, false)
    hierarchy.addObjects(cells)
    fireHierarchyUpdate()
    print "Converted. Cell objects now: " + getCellObjects().size()
}

def cells = getCellObjects()
if (cells.isEmpty()) { print "Conversion failed - no cell objects."; return }

// Every feature plugin below runs single-threaded. QuPath's default (47
// threads here) races on the shared MeasurementList and throws
// ConcurrentModificationException. Slower, but it completes.
def origThreads = PathPrefs.numCommandThreadsProperty().get()
PathPrefs.numCommandThreadsProperty().set(1)
print "Threads: " + origThreads + " -> 1 for feature computation"

try {

// ---------------- 2. SHAPE FEATURES ----------------
print "Shape features..."
// selectDetections() does not reliably pick up cell objects, which leaves
// the plugin with an empty selection and it silently does nothing.
selectObjects(cells)
addShapeMeasurements("AREA", "LENGTH", "CIRCULARITY", "SOLIDITY",
                     "MAX_DIAMETER", "MIN_DIAMETER")

// On CELL objects these are emitted per-ROI ("Nucleus: Area µm^2",
// "Cell: Area µm^2"); on plain detections they are unprefixed. Discover
// which, preferring the nucleus, since that is what the model was trained on.
def afterShape = new ArrayList(getCellObjects()[0].getMeasurementList().getMeasurementNames())
def areaName = afterShape.find { it == "Nucleus: Area µm^2" } ?:
               afterShape.find { it == "Area µm^2" } ?:
               afterShape.find { it.endsWith("Area µm^2") }
if (areaName == null) {
    print "  Shape features failed. Selected: " + getSelectedObjects().size()
    print "  All measurement names present:"
    new ArrayList(afterShape).sort().each { print "    " + it }
    return
}
def SHAPE_PREFIX = areaName.substring(0, areaName.length() - "Area µm^2".length())
print "  shape prefix: '" + SHAPE_PREFIX + "'  (from '" + areaName + "')"

// ---------------- 3. NUCLEAR INTENSITY ----------------
print "Nuclear intensity features..."
selectObjects(cells)
runPlugin('qupath.lib.algorithms.IntensityFeaturesPlugin',
    '{"pixelSizeMicrons":' + String.format("%.4f", mpp) + ',"region":"NUCLEUS",' +
    '"tileSizeMicrons":25.0,"colorOD":false,"colorStain1":true,"colorStain2":true,' +
    '"colorStain3":false,"colorRed":false,"colorGreen":false,"colorBlue":false,' +
    '"colorHue":false,"colorSaturation":false,"colorBrightness":false,' +
    '"doMean":true,"doStdDev":true,"doMinMax":true,"doMedian":true,' +
    '"doHaralick":false,"haralickDistance":1,"haralickBins":32}')

def afterIntensity = new ArrayList(getCellObjects()[0].getMeasurementList().getMeasurementNames())
// QuPath names nuclear intensity as "Nucleus: <px> µm per pixel: <stain>: <stat>",
// with the pixel size rounded to 2 decimals. Discover the exact prefix rather
// than assuming it - it varies with pixelSizeMicrons and QuPath version.
def NUC = afterIntensity.find { it.endsWith("Hematoxylin: Mean") &&
                                it.startsWith("Nucleus:") }
if (NUC == null) {
    print ""
    print "  Nuclear intensity NOT produced. Names containing Hematoxylin/DAB:"
    def found = afterIntensity.findAll { it.contains("Hematoxylin") || it.contains("DAB") }
    if (found.isEmpty()) {
        print "    (none at all - the intensity plugin did nothing)"
    } else {
        found.each { print "    " + it }
    }
    print ""
    print "  Cell objects: " + getCellObjects().size() +
          " | first has nucleus ROI: " +
          (getCellObjects()[0].getNucleusROI() != null)
    return
}
def NUC_PREFIX = NUC.substring(0, NUC.length() - "Hematoxylin: Mean".length())
print "  nuclear intensity prefix: '" + NUC_PREFIX + "'"

// ---------------- 4. SMOOTHED FEATURES ----------------
// Computed directly rather than via SmoothFeaturesPlugin: that plugin only
// accepts certain parent types, does nothing when detections sit under the
// root object, and races on the measurement lists. The maths is a
// Gaussian-weighted average over neighbouring nuclei, which is short.
print "Smoothed features (50um, 75um)..."

def baseNames = [
    SHAPE_PREFIX + "Area µm^2",       SHAPE_PREFIX + "Length µm",
    SHAPE_PREFIX + "Circularity",     SHAPE_PREFIX + "Solidity",
    SHAPE_PREFIX + "Max diameter µm", SHAPE_PREFIX + "Min diameter µm",
    NUC_PREFIX + "Hematoxylin: Mean",   NUC_PREFIX + "Hematoxylin: Median",
    NUC_PREFIX + "Hematoxylin: Min",    NUC_PREFIX + "Hematoxylin: Max",
    NUC_PREFIX + "Hematoxylin: Std.dev.",
    NUC_PREFIX + "DAB: Mean",           NUC_PREFIX + "DAB: Median",
    NUC_PREFIX + "DAB: Min",            NUC_PREFIX + "DAB: Max",
    NUC_PREFIX + "DAB: Std.dev."
]

int n = cells.size()
double[] cx = new double[n]
double[] cy = new double[n]
cells.eachWithIndex { c, i ->
    def r = c.getROI()
    cx[i] = r.getCentroidX() * mpp      // microns
    cy[i] = r.getCentroidY() * mpp
}
double[][] vals = new double[baseNames.size()][n]
baseNames.eachWithIndex { m, j ->
    cells.eachWithIndex { c, i ->
        def v = c.getMeasurementList().get(m)
        vals[j][i] = (v == null) ? Double.NaN : v
    }
}

// Grid index so each nucleus only compares against nearby ones, rather
// than all 90k+.
[50.0d, 75.0d].each { double fwhm ->
    double sigma = fwhm / 2.3548200450309493
    double radius = 2.0 * fwhm
    double cell = radius
    def grid = [:].withDefault { [] }
    for (int i = 0; i < n; i++) {
        grid[((int)Math.floor(cx[i]/cell)) + "_" + ((int)Math.floor(cy[i]/cell))] << i
    }

    String pfx = "Smoothed: " + (int)fwhm + " µm: "
    double[][] out = new double[baseNames.size()][n]
    double[] counts = new double[n]

    for (int i = 0; i < n; i++) {
        int gx = (int)Math.floor(cx[i]/cell), gy = (int)Math.floor(cy[i]/cell)
        double[] wsum = new double[baseNames.size()]
        double[] vsum = new double[baseNames.size()]
        int cnt = 0
        for (int ax = gx-1; ax <= gx+1; ax++) {
            for (int ay = gy-1; ay <= gy+1; ay++) {
                def bucket = grid[ax + "_" + ay]
                if (bucket.isEmpty()) continue
                for (int k : bucket) {
                    double dx = cx[k]-cx[i], dy = cy[k]-cy[i]
                    double d2 = dx*dx + dy*dy
                    if (d2 > radius*radius) continue
                    if (k != i) cnt++            // QuPath excludes self
                    double w = Math.exp(-d2 / (2.0*sigma*sigma))
                    for (int j = 0; j < baseNames.size(); j++) {
                        double v = vals[j][k]
                        if (!Double.isNaN(v)) { wsum[j] += w; vsum[j] += w*v }
                    }
                }
            }
        }
        counts[i] = cnt
        for (int j = 0; j < baseNames.size(); j++) {
            out[j][i] = (wsum[j] > 0) ? (vsum[j]/wsum[j]) : Double.NaN
        }
        if ((i+1) % 25000 == 0) print "    " + (int)fwhm + "um: " + (i+1) + "/" + n
    }

    cells.eachWithIndex { c, i ->
        def ml = c.getMeasurementList()
        for (int j = 0; j < baseNames.size(); j++) {
            ml.put(pfx + baseNames[j], out[j][i])
        }
        ml.put(pfx + "Nearby detection counts", counts[i])
    }
    print "  " + (int)fwhm + "um done"
}

def smoothNames = new ArrayList(getCellObjects()[0].getMeasurementList().getMeasurementNames())
print "  produced smoothed features: " + smoothNames.any { it.startsWith("Smoothed: 50") }

// ---------------- 5. REPORT ----------------
def names = new ArrayList(getCellObjects()[0].getMeasurementList().getMeasurementNames())
print ""
print "=== MEASUREMENT NAMES (" + names.size() + ") ==="
// getMeasurementNames() is immutable, so sort a copy.
names.sort().each { print "  " + it }

def expected = [
    SHAPE_PREFIX + "Area µm^2",         SHAPE_PREFIX + "Length µm",
    SHAPE_PREFIX + "Circularity",       SHAPE_PREFIX + "Solidity",
    SHAPE_PREFIX + "Max diameter µm",   SHAPE_PREFIX + "Min diameter µm",
    NUC_PREFIX + "Hematoxylin: Mean",   NUC_PREFIX + "Hematoxylin: Median",
    NUC_PREFIX + "Hematoxylin: Min",    NUC_PREFIX + "Hematoxylin: Max",
    NUC_PREFIX + "Hematoxylin: Std.dev.",
    NUC_PREFIX + "DAB: Mean",           NUC_PREFIX + "DAB: Median",
    NUC_PREFIX + "DAB: Min",            NUC_PREFIX + "DAB: Max",
    NUC_PREFIX + "DAB: Std.dev."
]
def allExpected = expected +
        expected.collect { "Smoothed: 50 µm: " + it } + ["Smoothed: 50 µm: Nearby detection counts"] +
        expected.collect { "Smoothed: 75 µm: " + it } + ["Smoothed: 75 µm: Nearby detection counts"]

def missing = allExpected.findAll { !names.contains(it) }
print ""
print "=== MATCH AGAINST CLASSIFIER'S 50 FEATURES ==="
print "Present: " + (allExpected.size() - missing.size()) + " / " + allExpected.size()
if (missing.isEmpty()) {
    print "All 50 present - ready for inference."
} else {
    print "Missing " + missing.size() + ":"
    missing.each { print "    " + it }
}

} finally {
    PathPrefs.numCommandThreadsProperty().set(origThreads)
    print "Threads restored to " + origThreads
}

fireHierarchyUpdate()
try {
    getProjectEntry()?.saveImageData(imageData)
    print "Saved."
} catch (Exception e) {
    print "Not saved: " + e.getMessage()
}
