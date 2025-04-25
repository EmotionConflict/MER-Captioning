#!/bin/bash

# Script to remove all .txt files from data/MER_test_subset/test_subset_au

# Check if the directory exists
if [ ! -d "data/MER_test_subset/test_subset_au" ]; then
    echo "Error: Directory data/MER_test_subset/test_subset_au does not exist."
    exit 1
fi

# Count the number of .txt files
txt_count=$(find data/MER_test_subset/test_subset_au -name "*.txt" | wc -l)
echo "Found $txt_count .txt files to remove."

# Remove all .txt files
find data/MER_test_subset/test_subset_au -name "*.txt" -exec rm {} \;

# Verify files were removed
remaining=$(find data/MER_test_subset/test_subset_au -name "*.txt" | wc -l)
echo "Removal complete. $remaining .txt files remaining." 