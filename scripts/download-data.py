import kagglehub

# Download latest version
path = kagglehub.dataset_download("gowrishankarp/newspaper-text-summarization-cnn-dailymail")

print("path to dataset files:", path)

# move the downloaded files and directories from path to the local save path
local_data_path = "./data"

print("copying files and directories from", path, "to", local_data_path)

import shutil
import os

def move_files_and_directories(src, dst):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        source = os.path.join(src, item)
        destin = os.path.join(dst, item)
        
        if os.path.isdir(source):
            shutil.copytree(source, destin, dirs_exist_ok=True)
        else:
            shutil.copy(source, destin)

move_files_and_directories(path, local_data_path)
