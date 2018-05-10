# DeCryptICS
# Deep Crypt Image Classifier and Segmenter

The tool is currently a work-in-progress.

Full functionality is expected to be present by June 2018.

External dependencies are currently:

* OpenSlide
* OpenCV
* TensorFlow / Keras
* PyVips

We are aiming to roll these dependencies into a Conda package alongside the DeCryptICS tool.

---

# Preparing input data

DeCryptICS currently takes in batches of .svs slide files stored within the folder structer: base\_path/batch\_ID/raw\_images/filename.svs

Output is saved in the folder structure: base\_path/batch\_ID/Analysed\_slides/Analysed\_filename/

Single/multiple images of standard types (.png, .jpg, .tiff, etc.) can also be analysed.

---

# Running the segmenter

---


