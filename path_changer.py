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

def change_paths(projfolder, newpath):
   projfile = projfolder + '/project.qpproj'
   groovyfile = projfolder + '/scripts/load_contours.groovy'

   # extract current file path from load contours script
   target = "def base_folder = "
   with open(groovyfile, 'r') as f:
      line = 1
      while line:
         line = f.readline()
         if (target in line):
            oldpath = line.split(target)[1].split('"')[1]
            break
   # replace in load contours file
   if '\\' in newpath:
      newpath = newpath.replace('\\', '\\\\')
      end_bool = True
      while end_bool:
         end_bool = newpath[-2:]=='\\\\'
         if end_bool: newpath = newpath[:-2]
      inplace_replace(groovyfile, oldpath, newpath+"\\\\Analysed_slides\\\\")
   if '/' in newpath:
      inplace_replace(groovyfile, oldpath, newpath+"/Analysed_slides/")

   # look for target in project file
   target = '"path":'
   with open(projfile, 'r') as f:
      line = 1
      while line:
         line = f.readline()
         if (target in line):
            oldpath = line.split(target)[1].split('"')[1]
            break

#   oldpath = oldpath.replace('//','/').split('Analysed_slides')[0]
   # replace in project file
   if '\\' in oldpath:
      oldpathlist = oldpath.split('\\')[:-1]
      sepp = '\\\\'
      sep = '\\'
      ## catch both examples (doesn't deal with a mixed \ \\ file path)
      oldpath1 = sepp.join([s for s in oldpathlist if s!='']) + sepp
      oldpath2 = sep.join([s for s in oldpathlist if s!='']) + sep
      if '\\' in newpath:
         inplace_replace(projfile, oldpath1, newpath + sepp)
         inplace_replace(projfile, oldpath2, newpath + sepp)
      elif '/' in newpath:
         inplace_replace(projfile, oldpath1, newpath + '/')
         inplace_replace(projfile, oldpath2, newpath + '/')
   if '/' in oldpath:
      oldpathlist = oldpath.split('/')[:-1]
      sepp = '/'
      oldpath1 = sepp.join([s for s in oldpathlist if s!='']) + sepp
      if oldpath1[0]!=sepp: oldpath1 = sepp + oldpath1
      if '\\' in newpath:
         inplace_replace(projfile, oldpath1, newpath + '\\\\')
      elif '/' in newpath:
         inplace_replace(projfile, oldpath1, newpath + '/')

def main():
   parser = argparse.ArgumentParser(description = "This script changes the image file paths hard-coded into the QuPath project files at runtime. ")

   parser.add_argument("path_to_qupath_project", help = "The full or relative path to the QuPath project folder containing project.qpproj")
   parser.add_argument("new_path_to_images", help = "The desired full path to the folder containing the relevant .svs image files ")

   # parse arguments
   args = parser.parse_args()
   projfolder = args.path_to_qupath_project # os.path.abspath(args.path_to_qupath_project)
   newpath = args.new_path_to_images # os.path.abspath(args.new_path_to_images)
   change_paths(projfolder, newpath)

if __name__=="__main__":
   main()
     
