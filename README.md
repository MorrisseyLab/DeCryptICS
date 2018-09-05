# DeCryptICS
# Deep Crypt Image Classifier and Segmenter

The tool is currently a work-in-progress.  Crypt counting is fully functional.

Install instructions in brief:

* Install Miniconda and create a conda environment with Python 3.6
* Run: conda install openslide opencv keras matplotlib scikit-learn numba joblib pandas numpy scipy libiconv ipython
* Run: pip install xlrd openslide-python
* Git clone this repository, https://github.com/MorrisseyLab/DeCryptICS.git
* Download the neural network weights from the Dropbox link listed in install\_instructions.txt and put them in ./DNN/weights/

And that's it! See the install\_instructions.txt file for a step-by-step (Linux) guide that may help, and for details on how to get the nerual network running on your GPU.

Main dependencies are currently:

* OpenSlide
* OpenCV
* TensorFlow / Keras
* ~~PyVips~~

We are aiming to roll these dependencies into a Conda package alongside the DeCryptICS tool. (GPU drivers and CUDA/CUDNN need to be installed separately.)

---

# Preparing input data

Run generate\_filepath\_list.py and point it at a folder containing the .svs slides to be analysed.

This will produce an input\_files.txt listing the paths to the slides (alternatively any list of remote or local file paths can be used).

---

# Running the segmenter

Run run\_script.py with an action and the path to the input file list.

View the help with "python run\_script.py -h" to see all available options.

Note: if running on a GPU and you receive an out-of-memory error, try reducing the "input\_size" in DNN/params.py (in nice powers of 2 so that downsampling always creates an integer size: 128, 256, 512, 1024...)

---

# Analysing the output

Manually curate a list of clones by running manual\_clone\_curation.py and pointing it at the output folder for a given slide (a folder of the form Analysed\_XXXXX). Again, "python manual\_clone\_curation.py -h" will show the help text.

Crypt, clone and clone patch counts can be found in /path/to/svs/slides/slide\_counts.csv and the clone counts will be updated when manual curation is performed using manual\_clone\_curation.py.

Contours for crypts, clones and patches can be loaded into QuPath using the automatically generated project script load\_contours.groovy within which a variable CLONE\_THRESHOLD can be tweaked to plot clones about which the algorithm was either more certain (CLONE\_THRESHOLD -> 1) or less certain (CLONE\_THRESHOLD -> 0.05). Setting a clone threshold of CLONE\_THRESHOLD = 0 will plot even those clones that were removed during manual curation. 
