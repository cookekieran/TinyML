"""
Data Organiser for Edge Impulse Image Exports

This script automates the sorting of raw training images into a directory 
structure suitable for ImageFolder loaders, like Pytorch.

Logic:
    It parses filenames using the first dot-separated segment as the label 
    (e.g., 'rock.01.jpg' -> creates a folder named 'rock').

Directory Structure:
    Input:  [data]/label.unique_id.jpg
    Output: [data]/label/label.unique_id.jpg
"""

import os
import shutil

source_folder = './data/training'

def organize_data():
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    
    count = 0
    for filename in files:
        if filename.endswith('.py'):
            continue
            
        parts = filename.split('.')
        category = parts[0]
        category_path = os.path.join(source_folder, category)
        if not os.path.exists(category_path):
            os.makedirs(category_path)
            
        src = os.path.join(source_folder, filename)
        dst = os.path.join(category_path, filename)
        
        try:
            shutil.move(src, dst)
            count += 1
        except Exception as e:
            print(f"Error moving {filename}: {e}")

    print(f"Organised {count} files into their respective folders.")

if __name__ == "__main__":
    organize_data()