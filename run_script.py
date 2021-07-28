#!/usr/bin/env python3
import glob, os, sys
import numpy  as np
import pandas as pd
import argparse, datetime
from pathlib import Path
from qupath_project import create_qupath_project
from MiscFunctions  import mkdir_p, process_input_file

def run_analysis(args):
   ## check args
   print("Running with the following inputs:")
   print('action       = {!r}'.format(args.action))
   print('input_file   = {!r}'.format(args.input_file))
   print('qp_proj_name = {!r}'.format(args.qp_proj_name))
   print('force_repeat = {!r}'.format(args.r))
#   print('mouse        = {!r}'.format(args.mouse))
   print('\n...Working...\n')

   ## Find output folder, create directory structures
   base_path, folder_out, file_in, folders_to_analyse, full_paths = process_input_file(args.input_file)
   mkdir_p(folder_out)
   for fta in folders_to_analyse: mkdir_p(fta)
         
   ## Create QuPath project for the current batch
   decryptics_folder = os.path.dirname(os.path.abspath(__file__))
   qupath_project_path = base_path + '/' + args.qp_proj_name
   create_qupath_project(qupath_project_path, full_paths, file_in, folder_out, decryptics_folder)
   print('QuPath project created in %s' % qupath_project_path)

   if args.action=='count':
      from whole_slide_run_funcs import run_slide
      
      ## run model
      for i in range(len(file_in)):
         if (os.path.isfile(folders_to_analyse[i]+'raw_crypt_output.csv') and args.r==False):         
            print('Passing on %s, image previously analysed.' % full_paths[i])
            pass
         else:
            print('Beginning work on %s.' % full_paths[i])
            try:
               run_slide(full_paths[i], folders_to_analyse[i])
            except:
               print('Passing due to error.')
                    
if __name__=='__main__':
   parser = argparse.ArgumentParser(description = "This script takes as input a list of full paths (local or remote) to .svs files, "
                                                   "or a larger dataframe that contains such a list as a column. "
                                                   "The output is a QuPath project containing all .svs slides. "
                                                   "Optionally, the .svs slides can be analysed to count crypts and clones, "
                                                   "or crypts only; detected contours are added to the QuPath project. " )
    
   parser.add_argument("action", choices = ["read" , "count"], 
                                 default = "read", 
                                 help = "Action to carry out. "
                                        "read: create QuPath project for input slides; "
                                        "count: counts crypts (and possibly clones) and creates QuPath project. ")

   parser.add_argument("input_file", help = "A file containing a list of full paths to slides (or array with column acting as said list). ")

   parser.add_argument('-q', action = "store",
                        dest = "qp_proj_name", 
                        default = "qupath_project_"+datetime.datetime.now().strftime("%d-%m-%Y_%H-%M"),
                        help = "Optionally set the name of the QuPath project folder to be created. "
                               "Defaults to 'qupath_project_DATE_TIME'. ")

   parser.add_argument('-r', action = "store_true", 
                             default = False,
                             help = "Forces repeat analysis of slides with existing crypt contour files. "
                                    "Defaults to False if -r flag missing. ")

#   parser.add_argument('-mouse', action = "store_true", 
#                                 default = False,
#                                 help    = "Indicates we are analysing mouse tissue. ")
                                 
   args = parser.parse_args()
   run_analysis(args)
        
