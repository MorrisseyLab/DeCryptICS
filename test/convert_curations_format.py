import glob
import numpy as np
import pandas as pd
from read_svs_class import load_data_preferentially
from sklearn.neighbors import NearestNeighbors

# ed's curation
already_curated = pd.read_csv("manual_curation_files/curated_files_summary.txt", 
                              names = ["file_name", "slide_crtd"])
already_curated = already_curated[already_curated['slide_crtd'] != "cancel"]

# my curation
imgpaths = glob.glob("/home/doran/Work/images/Leeds_May2019/curated_cryptdata/train/KM*/*.svs")
dontuse = pd.read_csv('/home/doran/Work/py_code/experimental_DeCryptICS/DNN/fullres_unet/drop_slides_from_mutfrac.csv')
imgpaths = list(np.setdiff1d(imgpaths, dontuse))
file_paths_filt = pd.DataFrame({'path':imgpaths})
# filter down to those curated for partials
impath_pd = file_paths_filt.sample(250, random_state=222)
partial_curation = pd.read_csv('/home/doran/Work/py_code/new_DeCryptICS/partial_scoring.csv')
curatedpaths = pd.DataFrame({'path':list(partial_curation['path'].drop_duplicates(keep='first')) +
                                    list(impath_pd['path'])}).drop_duplicates(keep='first')
# remove poorly stained slides or radiotherapy patients
bad_slides = pd.read_csv('/home/doran/Work/py_code/new_DeCryptICS/slidequality_scoring.csv')
radiother = np.where(bad_slides['quality_label']==2)[0]
staining = np.where(bad_slides['quality_label']==0)[0]
dontuse2 = np.asarray(bad_slides['path'])[np.hstack([radiother, staining])]
dontuse2 = pd.DataFrame({'path':list(dontuse2)}).drop_duplicates(keep='first')
curatedpaths = pd.concat([curatedpaths, dontuse2, dontuse2]).drop_duplicates(keep=False)

# take the new ones in my list and resave in ed's format
folder_out = '/home/doran/Work/py_code/new_DeCryptICS/newfiles/manual_curation_files/'
new_curations = np.setdiff1d(curatedpaths['path'], already_curated['file_name'])
partial_curation = np.asarray(partial_curation)
for imgpath in new_curations:
   ## load data and contours
   sld_dat, mark = load_data_preferentially(imgpath)
   ## ammend sld_dat with partial/FP curation
   ammend = partial_curation[np.where(partial_curation[:,0]==imgpath)[0], :]
   if ammend.shape[0]>0:
      sld_dat = np.hstack([sld_dat, np.zeros((sld_dat.shape[0],1))])
      theseclones = sld_dat[np.where(sld_dat[:,3]>0)[0],:]
      nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(theseclones[:,:2])
      distances, indices = nbrs.kneighbors(ammend[:,1:3])
      # this assumes clones and partials are the last two colums of sld_dat
      sld_dat[np.where(sld_dat[:,3]>0)[0], -2:] = ammend[indices[:,0], -2:].astype(np.float32)
   sld_dat[np.where(sld_dat[:,-1]>0)[0], 3] = 2
   file_out_cur = imgpath.split("/")[-1].replace(".svs", "")
   file_out_cur = folder_out + file_out_cur + "_curtd.csv"
   pd.DataFrame(sld_dat[:, 0:4]).to_csv(file_out_cur)
   with open(folder_out + 'curated_files_summary.txt', 'a') as file:
      file.write(imgpath + "," + file_out_cur + "\n")


