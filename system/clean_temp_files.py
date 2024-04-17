import shutil
import sys
  
directory_path = sys.argv[1]
  
# Forcefully delete the directory and its contents
try:
    shutil.rmtree(directory_path)
    print(f'{directory_path} deleted.')
except:
    print(f'{directory_path} already deleted.')