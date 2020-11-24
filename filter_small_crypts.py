import numpy as np
import matplotlib.pyplot as plt
from MiscFunctions import read_cnt_text_file, write_cnt_text_file
import argparse

def not_inplace_replace(file_in, file_out, old_string, new_string):
   replace = True
   with open(file_in, 'r') as f:
      s = f.read()
      if old_string not in s:
         print('%s not found in %s.' % (old_string, file_in))
         replace = False

   if replace==True:
      # Safely write the changed content, if found in the file
      s = s.replace(old_string, new_string)
      with open(file_out, 'w') as f:
         print('Changing %s to %s in new file %s' % (old_string, new_string, file_out))
         f.write(s)

def main():
   parser = argparse.ArgumentParser(description = "Filter out tiny noise crypts." )

   parser.add_argument("folder_to_analyse", help = "The full or relative path to the Analysed_XXXXX folder containing crypt/clone contours, "
                                                   "clone scores, patch sizes and crypt_network_data for the slide XXXXX.svs. ")

   parser.add_argument('-s', action = "store",
                        dest = "threshold_size", 
                        default = 0,
                        help = "Set the size of the crypts to be filtered. ")
                                                                              
                                                   
   args = parser.parse_args()
   folder_to_analyse = args.folder_to_analyse
   threshold_size = int(args.threshold_size)
   
   ## read contour network file
   data = np.loadtxt(folder_to_analyse + '/crypt_network_data.txt')
   numcrypts = data.shape[0]
   areas = data[:,6]
   
   ## plot histogram of crypt areas
   if (numcrypts>0):
      plt.hist(areas, bins = int(numcrypts/50))
      plt.show()
   else:
      print('No crypts found in slide. Exiting.')
      return 1
   
   ## if threshold_size is >0 read contours, filter them, output reduced set
   if (threshold_size>0 and numcrypts>0):
      cnts = read_cnt_text_file(folder_to_analyse + '/crypt_contours.txt')
      if len(cnts)!=numcrypts:
         print('Error: Number of contours does not equals size of data! Exiting.')
         return 1
      goodinds = np.where(areas>threshold_size)[0]
      newcnts = [cnts[i] for i in goodinds]
      write_cnt_text_file(newcnts, folder_to_analyse + '/reduced_crypt_contours.txt')
      
      ## output new groovy script to load reduced contours
      groovyfile_old = folder_to_analyse.split('Analysed_slides')[0]+'block_analysis/scripts/load_contours.groovy'
      groovyfile_new = folder_to_analyse.split('Analysed_slides')[0]+'block_analysis/scripts/load_reduced_contours.groovy'
      not_inplace_replace(groovyfile_old, groovyfile_new, 'crypt_contours.txt', 'reduced_crypt_contours.txt')
      
if __name__=="__main__":
   main()
      
      
      
      
