### Move M4Singer snippets to a collection folder

import os
import shutil

# Directory where the .wav files are located
source_dir = '/mntnfs/lee_data1/zhangxueyao/dataset/M4Singer'

# Directory where the copied .wav files will be placed
target_dir = '/mntnfs/lee_data1/haoyan/dataset/M4Singer/collection'

# Ensure the target directory exists
os.makedirs(target_dir, exist_ok=True)

# Iterate over the source directory and its subdirectories
for root, dirs, files in os.walk(source_dir):
    for file in files:
        # Check if the file is a .wav file
        if file.endswith('.wav'):
            # Get the original folder name
            original_folder_name = os.path.basename(root)

            # Create the new file name
            new_file_name = f"{original_folder_name}_{file}"

            # Get the full path of the source file
            source_file_path = os.path.join(root, file)

            # Get the full path of the target file
            target_file_path = os.path.join(target_dir, new_file_name)

            # Copy the file
            shutil.copy2(source_file_path, target_file_path)

print("Copying process finished.")
