//guiscript=true
import qupath.lib.objects.*
import qupath.lib.roi.*
import qupath.lib.objects.classes.PathClass;
import qupath.lib.common.ColorTools;
import qupath.lib.objects.classes.PathClassFactory;
import java.io.File;

double THRESHOLD = 0.5

// Some code taken from:
// https://groups.google.com/forum/#!topic/qupath-users/j_Wd1hy4eKM
// https://groups.google.com/forum/#!topic/qupath-users/QyzvMjQ08cY
// https://github.com/qupath/qupath/issues/169

// Add object classes
def CryptClass   = PathClassFactory.getPathClass("Crypt")
def CloneClass   = PathClassFactory.getPathClass("Clone")
def PartialClass = PathClassFactory.getPathClass("Partial")
def FufiClass    = PathClassFactory.getPathClass("Fufi")
def PatchClass = PathClassFactory.getPathClass("Patch")

// clean and remake path classes
def pathClasses  = getQuPath().getAvailablePathClasses()
pathClasses.remove(CryptClass)
pathClasses.remove(CloneClass)
pathClasses.remove(PartialClass)
pathClasses.remove(FufiClass)
pathClasses.remove(PatchClass)
if (!pathClasses.contains(CryptClass)) {
	pathClasses.add(CryptClass)
}
if (!pathClasses.contains(CloneClass)) {
	pathClasses.add(CloneClass)
}
if (!pathClasses.contains(PartialClass)) {
	pathClasses.add(PartialClass)
}
if (!pathClasses.contains(FufiClass)) {
	pathClasses.add(FufiClass)
}
if (!pathClasses.contains(PatchClass)) {
	pathClasses.add(PatchClass)
}
PathClassFactory.getPathClass("Crypt").setColor(ColorTools.makeRGB(175,0,0))
PathClassFactory.getPathClass("Clone").setColor(ColorTools.makeRGB(10,152,30))
PathClassFactory.getPathClass("Partial").setColor(ColorTools.makeRGB(10,15,135))
PathClassFactory.getPathClass("Fufi").setColor(ColorTools.makeRGB(120,5,100))
PathClassFactory.getPathClass("Patch").setColor(ColorTools.makeRGB(170,110,0))

// Define folder structure
def base_folder = BASEPATH_PLACEHOLDER
def cur_file = getCurrentImageData().getServer().getPath()
print cur_file
def ff = cur_file.tokenize(File.separator)[-1].tokenize('.')[0]

// Add crypt contours and patch contours
def file1       = new File(base_folder+"Analysed_"+ff+"/crypt_contours.txt")
def file2       = new File(base_folder+"Analysed_"+ff+"/patch_contours.txt")
//def filescores = new File(base_folder+"Analysed_"+ff+"/crypt_mask.txt")
def filescores1 = new File(base_folder+"Analysed_"+ff+"/clone_mask.txt")
def filescores2 = new File(base_folder+"Analysed_"+ff+"/partial_mask.txt")
def filescores3 = new File(base_folder+"Analysed_"+ff+"/fufi_mask.txt")
if( file1.exists() ) {
	def lines1 = file1.readLines()
	num_rois = lines1.size/2
	def pathObjects = []
	def pathObjects1 = []
	def pathObjects2 = []
	def pathObjects3 = []
	//def linesscores = filescores.readLines()
	def linesscores1 = filescores1.readLines()
	def linesscores2 = filescores2.readLines()
	def linesscores3 = filescores3.readLines()
	for (i = 0; i<num_rois; i++) {
		float[] x1 = lines1[2*i].tokenize(',') as float[]
		float[] y1 = lines1[2*i+1].tokenize(',') as float[]
      //double score  = linesscores[i] as double
		double score = 1.0
		double score1 = linesscores1[i] as double
		double score2 = linesscores2[i] as double
		double score3 = linesscores3[i] as double
		if ( score >= THRESHOLD ) {
		   def roi = new PolygonROI(x1, y1, -300, 0, 0)
		   pathObjects << new PathDetectionObject(roi, CryptClass)
		}		
		if ( score1 >= THRESHOLD ) {
     		def roi1 = new PolygonROI(x1, y1, -300, 0, 0)
			pathObjects1 << new PathAnnotationObject(roi1, CloneClass)
		}
		if ( score2 >= THRESHOLD ) {
         def roi2 = new PolygonROI(x1, y1, -300, 0, 0)
         pathObjects2 << new PathAnnotationObject(roi2, PartialClass)
		}
		if ( score3 >= THRESHOLD ) {
         def roi3 = new PolygonROI(x1, y1, -300, 0, 0)
         pathObjects3 << new PathAnnotationObject(roi3, FufiClass)
		}
	}
	addObjects(pathObjects)
	addObjects(pathObjects1)
	addObjects(pathObjects2)
	addObjects(pathObjects3)
	print("Done crypts!")
}
if( file2.exists() ) {
	def lines2 = file2.readLines()
	num_rois = lines2.size/2
	def pathObjects4 = []
	for (i = 0; i<num_rois; i++) {
		float[] x2 = lines2[2*i].tokenize(',') as float[]
		float[] y2 = lines2[2*i+1].tokenize(',') as float[]
	   def roi4 = new PolygonROI(x2, y2, -300, 0, 0)
	   pathObjects4 << new PathAnnotationObject(roi4, PatchClass)
	}
	addObjects(pathObjects4)
	print("Done patches!")
}
