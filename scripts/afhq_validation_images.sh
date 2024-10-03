#!/bin/bash

# Directory where the validation images will be moved
val_dir="./data/afhq_cat/val/cat"
# Directory where the validation list file is saved
output_dir="./data/afhq_cat"
# File containing the names of the selected images
val_list_file="splits/afhq_cat/validation_images.txt"

# Ensure the destination directory exists
mkdir -p "$val_dir"

# Move the selected images to the validation directory
while IFS= read -r image_path; do
    # Check if the file exists before moving
    if [ -f "$image_path" ]; then
        mv "$image_path" "$val_dir"
        echo "Moved: $image_path"
    else
        echo "File not found: $image_path"
    fi
done < "$val_list_file"

echo "Images have been moved to $val_dir."
