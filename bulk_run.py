## run bulk blocks
from run_script import run_analysis
import datetime
import glob

class Args():
   def __init__(self, input_file, action='read', qp_proj_name=None, r=False):
      self.action = action
      self.input_file = input_file   
      self.r = r
      if qp_proj_name is None:
         self.qp_proj_name = 'qp_project_'+datetime.datetime.now().strftime('%d-%m-%Y_%H-%M')
      else:
         self.qp_proj_name = qp_proj_name      
         

# choose block sets to run in bulk
imgbase = '/home/doran/Work/images/'
blocks = glob.glob(imgbase + 'Blocks/block*/input_files.txt')
kras = glob.glob(imgbase + 'KRAS_study/block*/input_files.txt')
newserials = glob.glob(imgbase + 'Serial_blocks_Oct2019/block*/input_files.txt')
run_set = blocks + kras + newserials

for block in run_set:
   args = Args(block, action='count', qp_proj_name='block_analysis_v2', r=False)
   run_analysis(args)

