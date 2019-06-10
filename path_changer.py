#!/usr/bin/env python3

import sys, os

if __name__=="__main__":
   replace = True
   projfile = os.path.abspath(sys.argv[1])
   with open(projfile) as f:
      s = f.read()
      if sys.argv[2] not in s:
         print('%s not found in %s.' % (sys.argv[2], projfile))
         replace = False

   if replace==True:
      # Safely write the changed content, if found in the file
      s = s.replace(sys.argv[2], sys.argv[3])
      with open(projfile, 'w') as f:
         print('Changing %s to %s in %s' % (sys.argv[2], sys.argv[3], projfile))
         f.write(s)  
     
