#!/usr/bin/env python3

import sys, os, argparse

def inplace_replace(file_in, old_string, new_string):
   replace = True
   with open(file_in) as f:
      s = f.read()
      if old_string not in s:
         print('%s not found in %s.' % (old_string, file_in))
         replace = False

   if replace==True:
      # Safely write the changed content, if found in the file
      s = s.replace(old_string, new_string)
      with open(file_in, 'w') as f:
         print('Changing %s to %s in %s' % (old_string, new_string, file_in))
         f.write(s)


def main():
   parser = argparse.ArgumentParser(description = "This script changes the image file paths hard-coded into the QuPath project files at runtime. ")

   parser.add_argument("path_to_qupath_project", help = "The full or relative path to the QuPath project folder containing project.qpproj")
   parser.add_argument("new_path_to_images", help = "The desired full path to the folder containing the relevant .svs image files ")

   # parse arguments
   args = parser.parse_args()
   projfolder = os.path.abspath(args.path_to_qupath_project)
   newpath = os.path.abspath(args.new_path_to_images)
   projfile = projfolder + '/project.qpproj'
   groovyfile = projfolder + '/scripts/load_contours.groovy'

   # extract current file path from load contours script
   target = "def base_folder = "
   with open(groovyfile, 'r') as f:
      line = f.readline()
      if (target in line):
         oldpath = line.split(target)[0].split('"')[-1]
         line = None
      while line:
         line = f.readline()
         if (target in line):
            oldpath = line.split(target)[1].split('"')[1]
            break
   # replace in load contours file
   inplace_replace(groovyfile, oldpath, newpath)

   oldpath = oldpath.replace('//','/').split('Analysed_slides')[0]
   # replace in project file
   inplace_replace(projfile, oldpath, newpath)

if __name__=="__main__":
   main()
     
