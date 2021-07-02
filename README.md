# DeCryptICS
# Deep Crypt Image Classifier and Segmenter

This is a tool for segmenting and classifying intestinal glands on IHC and H&E slides. The functionality:
* crypts will be segmented and counted
* mutations/perturbations that cause loss of nuclear or cytoplasmic staining will be classified as full-crypt clones or partial-crypt clones
* multi-crypt clones will be organised into clonal patches and patch sizes will be recorded
* fusion/fission (fufi) events will be classified and counted

Install instructions in brief:

* Install Miniconda, add the bioconda, conda-forge and r channels and create a conda environment with Python 3.8
* conda install openslide opencv matplotlib scikit-learn joblib pandas numpy scipy libiconv ipython scikit-image pyvips 
* pip install xlrd openslide-python networkx community albumentations tensorflow==2.4 tensorflow-addons==0.13 
* Git clone this repository, https://github.com/MorrisseyLab/DeCryptICS.git
* Download the neural network weights from the Dropbox link listed in install\_instructions.txt and put them in ./weights/

And that's it! See https://www.tensorflow.org/install/gpu for a instructions to get your GPU working.

Main dependencies are currently:

* OpenSlide
* OpenCV
* TensorFlow / Keras
* PyVips

---

# Preparing input data

Run generate\_block\_list.py and point it at a folder containing the .svs slides to be analysed. If all slides in the folder use the same clonal mark/stain, denote the mark to be used with the flag -c and then the name or number of the mark:

python generate\_block\_list.py /full/path/to/images/ -c KDM6A

or

python generate\_block\_list.py /full/path/to/images/ -c 1

If the folder contains slides with different stainings, for instance if they are different sections of the same block with different treatments, create a file called "slide\_info.csv" with the columns "Image ID" and "mark", denoting the filename (with or without file extension) and the clonal mark used (name or number). Then run generate\_block\_list.py without the -c flag:

python generate\_block\_list.py /full/path/to/block/

This will produce an input\_files.txt listing the paths to the slides and the mark to be used in clone finding.

---

# Running the segmenter

Run run\_script.py with an action and the path to the input file list. For example,

python run\_script.py count /full/path/to/block/input\_files.py

View the help with "python run\_script.py -h" to see all available options.

# Processing the raw output

Raw output is given in terms of probabilities of each bounding box being each object type (clone, partial, fufi). The processing step threshold these probabilities either with constant thresholds or manually with a slider by passing the -m flag:

python process\_output.py /full/path/to/block/input\_files.py -m

Thresholds can be set for non-manual processing as follows:

python process\_output.py /full/path/to/block/input\_files.py -c 0.5 -p 0.5 -f 0.5

for -c, -p and -f being clone, partial and fufi, respectively.

---

# Analysing the output

Manually curate a list of clones by running manual\_clone\_curation.py and pointing it at the output folder for a given slide (a folder of the form Analysed\_XXXXX). Again, "python manual\_clone\_curation.py -h" will show the help text.

Crypt, mutant crypt, clone and clone patch counts can be found in /path/to/svs/slides/slide\_counts.csv. The clone counts will be updated when manual curation is performed using manual\_clone\_curation.py.  The "NMutantCrypts" column of slides\_counts.csv counts the number of individual mutant crypts; "NClones" counts patches of mutant crypts as a single clone; "NPatches" counts the number of mutant patches containing two or more crypts.

Contours for crypts, clones and patches can be loaded into QuPath using the automatically generated project script load\_contours.groovy within which a variable CLONE\_THRESHOLD dictates which clone contours to plot. The clone\_score.txt file contains the score values for clone contours, which are updated when performing manual curation so that false positives can be hidden.
