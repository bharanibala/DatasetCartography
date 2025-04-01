import os
import shutil
import pandas as pd

# Input CSV file and base output directory
input_csv = "/home/b/bharanibala/noisefind/aumrank/aum-master/vgg/nepal_balanced_resnet50_threshold95_weightnone/clean_images.csv"  # Replace with your CSV file path
output_base_dir = "/home/b/bharanibala/noisefind/aumrank/aum-master/vgg/nepal_balanced/aum_notpretrained"  # Define a new base directory

# Ensure the base output directory exists
os.makedirs(output_base_dir, exist_ok=True)

# Read the CSV file into a DataFrame
df = pd.read_csv(input_csv, header=None, names=["image_path"])

# Function to organize files
def organize_files(df, output_base_dir):
    for _, row in df.iterrows():
        file_path = row['image_path']
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        
        # Parse sub-directory names based on the path structure
        sub_dir = os.path.relpath(file_path, "/home/b/bharanibala/noisefind/aumrank/aum-master/vgg/nepal_balanced").split(os.sep)[0:2]
        target_dir = os.path.join(output_base_dir, *sub_dir)

        # Ensure the target directory exists
        os.makedirs(target_dir, exist_ok=True)

        # Move the file to the new directory
        shutil.copy(file_path, target_dir)  # Use `shutil.move` if you want to move instead of copy
        print(f"Copied: {file_path} to {target_dir}")

# Run the organization process
organize_files(df, output_base_dir)

print(f"Files have been organized into {output_base_dir}")