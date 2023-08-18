import shutil
  
directory_path = 'temp'
  
# Forcefully delete the directory and its contents
try:
    shutil.rmtree(directory_path)
    print('Deleted.')
except:
    print('Already deleted.')