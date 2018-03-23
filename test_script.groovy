import qupath.lib.objects.*
import qupath.lib.roi.*

// Some code taken from here https://groups.google.com/forum/#!topic/qupath-users/j_Wd1hy4eKM


print getCurrentImageData().getServer().getPath()


// This runs for img_mPAS_devel_PurpleClone
//def my_path_class = new qupath.lib.objects.classes.PathClassFactory.getPathClass("Crypts", qupath.lib.common.ColorTools.makeRGB(0, 255, 0)) // r, g, b
def file1 = new File('/home/doran/pythonOpenCV_forDoran/test_cnt.txt')
def lines1 = file1.readLines()

num_rois = lines1.size/2
def pathObjects = []
for (i = 0; i <num_rois; i++) {
    float[] x1 = lines1[2*i].tokenize(',') as float[]
    float[] y1 = lines1[2*i+1].tokenize(',') as float[]    
    // Create object
    def roi = new PolygonROI(x1, y1, -1, 0, 0)
    //def pathObject = new PathDetectionObject(roi)
    pathObjects << new PathDetectionObject(roi)
    //def pathObject = new PathAnnotationObject(roi)
    // Add object to hierarchy
}
addObjects(pathObjects) //, my_path_class)
print("Done!")



def file = new File('/home/doran/pythonOpenCV_forDoran/test_mPAS_cnt.txt')
def lines = file.readLines()
def pathObjects2 = []
num_rois = lines.size/2
for (i = 0; i <num_rois; i++) {
    float[] x1 = lines[2*i].tokenize(',') as float[]
    float[] y1 = lines[2*i+1].tokenize(',') as float[]    
    // Create object
    def roi = new PolygonROI(x1, y1, -1, 0, 0)
    //def pathObject = new PathDetectionObject(roi)
    pathObjects2 << new PathDetectionObject(roi)
    // Add object to hierarchy
//    addObject(pathObject) //, my_path_class)
}
addObjects(pathObjects2) //, my_path_class)
print("Done!")



