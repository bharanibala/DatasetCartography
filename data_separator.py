import pandas as pd
import ast

# Input and output CSV file paths
input_csv = "/home/b/bharanibala/noisefind/carto/cartography-main/filtered/cartography_variability_0.33/NepalBalanced/filtered_images.csv"
output_csv = "/home/b/bharanibala/noisefind/carto/cartography-main/filtered/cartography_variability_0.33/NepalBalanced/output.csv"

import re

def extract_image_path(label_str):
    try:
        # Regex to match the value of image_path
        match = re.search(r"'image_path':\s*'(.*?)'", label_str)
        if match:
            return match.group(1)
        else:
            print(f"No image_path found in: {label_str}")
            return None
    except Exception as e:
        print(f"Error extracting image_path: {e}")
        return None

# Load the CSV, process, and save the output
def process_csv(input_csv, output_csv):
    # Load the input CSV into a DataFrame
    df = pd.read_csv(input_csv)

    # Extract image_path using the function
    df['image_path'] = df['label'].apply(extract_image_path)

    # Save only the image_path column to the output CSV
    df[['image_path']].dropna().to_csv(output_csv, index=False)
    print(f"Image paths have been extracted and saved to {output_csv}")

# Run the script
process_csv(input_csv, output_csv)