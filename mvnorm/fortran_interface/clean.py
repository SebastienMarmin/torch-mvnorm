# clean compilation fossils
from shutil import rmtree
from os import remove
import glob
 
try:
      rmtree("./temp/")
except OSError:
      pass
try:
      rmtree("./build/")
except OSError:
      pass
fileList = glob.glob('*gfunc*.mod*')
print(fileList)
for filename in fileList:
      try:
          remove(filename)
      except OSError:
          print("Error while deleting file")
