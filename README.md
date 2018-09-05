# DeCryptICS
# Deep Crypt Image Classifier and Segmenter

The tool is currently a work-in-progress.  Crypt counting is fully functional.

See the install\_instructions.txt file for a step-by-step (Linux) guide that may help.

Main dependencies are currently:

* OpenSlide
* OpenCV
* TensorFlow / Keras
* ~~PyVips~~

We are aiming to roll these dependencies into a Conda package alongside the DeCryptICS tool. (GPU drivers and CUDA/CUDNN need to be installed separately, see install\_instructions.txt.)

---

# Preparing input data

Run generate\_filepath\_list.py and point it at a folder containing the .svs slides to be analysed.

This will produce an input\_files.txt listing the paths to the slides (alternatively any list of remote or local file paths can be used).

---

# Running the segmenter

Run run\_script.py with an action and the path to the input file list.

View the help with "python run\_script.py -h" to see all available options.

Note: if running on a GPU and you receive an out-of-memory error, try reducing the "input\_size" in DNN/params.py (in nice powers of 2 so that downsampling always creates an integer size: 128, 256, 512, 1024...)

Manually curate a list of clones by running manual\_clone\_curation.py and pointing it at the output folder for a given slide (a folder of the form Analysed\_XXXXX).

Again, "python manual\_clone\_curation.py -h" will show the help text.

---


