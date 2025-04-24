#!/usr/bin/env python3
import os
import csv
import random
import shutil
import pandas as pd
from collections import Counter

# Create necessary directories
os.makedirs('10_per_emotion_experiment', exist_ok=True)
os.makedirs('10_per_emotion_experiment/videos-mp4', exist_ok=True)
os.makedirs('10_per_emotion_experiment/videos-avi', exist_ok=True)

# Process both test1 and test2 datasets
test_sets = [
    {
        'label_file': 'mer2023-test-labels/test1-label.csv',
        'video_dir': 'mer2023test1&2/test1',
        'video_ext': '.avi',
        'video_destination_dir': '10_per_emotion_experiment/videos-avi',
        "file_output_name": "random_10_per_emotion_test1.csv"
    },
    {
        'label_file': 'mer2023-test-labels/test2-label.csv',
        'video_dir': 'mer2023test1&2/test2',
        'video_ext': '.mp4',
        'video_destination_dir': '10_per_emotion_experiment/videos-mp4',
        "file_output_name": "random_10_per_emotion_test2.csv"
    }
]

all_emotions = set()
all_samples = []
dataset_samples = {}

for test_set in test_sets:
    label_file = test_set['label_file']
    video_dir = test_set['video_dir']
    video_ext = test_set['video_ext']
    file_output_name = test_set['file_output_name']
    video_destination_dir = test_set['video_destination_dir']
    
    print(f"\nProcessing {label_file}...")
    
    # Check if the label file exists
    if not os.path.exists(label_file):
        print(f"Warning: {label_file} does not exist. Skipping.")
        continue
    
    # Read the CSV file
    df = pd.read_csv(label_file)
    
    # Add source information to each row for later use (but won't be included in output CSV)
    df['source_dir'] = video_dir
    df['video_ext'] = video_ext
    df['destination_dir'] = video_destination_dir
    
    # Count emotions in this dataset
    emotion_counts = Counter(df['discrete'])
    print(f"Emotions in {os.path.basename(label_file)}:")
    for emotion, count in emotion_counts.items():
        print(f"  {emotion}: {count}")
    print(f"Total unique emotions in this set: {len(emotion_counts)}")
    
    # Update the set of all emotions
    all_emotions.update(emotion_counts.keys())
    
    # Add to all samples list
    all_samples.append(df)
    
    # Process individual dataset samples
    random_samples_dataset = []
    for emotion in emotion_counts.keys():
        # Get all rows for this emotion
        emotion_rows = df[df['discrete'] == emotion]
        
        # If there are at least 10 samples, randomly select 10
        num_samples = min(10, len(emotion_rows))
        selected_rows = emotion_rows.sample(n=num_samples, random_state=42)
        
        # Add to our list of random samples
        random_samples_dataset.append(selected_rows)
    
    # Combine all selected rows into one DataFrame for this dataset
    if random_samples_dataset:
        output_df_dataset = pd.concat(random_samples_dataset)
        
        # Make a copy of the dataframe for CSV output without the extra columns
        output_csv_df = output_df_dataset.drop(columns=['source_dir', 'video_ext', 'destination_dir'])
        
        # Save to CSV
        output_file_dataset = os.path.join('10_per_emotion_experiment', file_output_name)
        output_csv_df.to_csv(output_file_dataset, index=False)
        print(f"Saved {len(output_df_dataset)} samples from {os.path.basename(label_file)} to {output_file_dataset}")
        
        # Store for later use in video copying (keeping the processing columns)
        dataset_samples[file_output_name] = output_df_dataset

# Combine datasets
combined_df = pd.concat(all_samples, ignore_index=True)

# Task 1: Show total counts across all datasets
combined_emotion_counts = Counter(combined_df['discrete'])
print("\nCombined emotions across all datasets:")
for emotion, count in combined_emotion_counts.items():
    print(f"  {emotion}: {count}")
print(f"Total unique emotions across all datasets: {len(combined_emotion_counts)}")

# Task 2: Randomly select 10 samples from each emotion for combined dataset
random_samples = []
for emotion in combined_emotion_counts.keys():
    # Get all rows for this emotion
    emotion_rows = combined_df[combined_df['discrete'] == emotion]
    
    # If there are at least 10 samples, randomly select 10
    num_samples = min(10, len(emotion_rows))
    selected_rows = emotion_rows.sample(n=num_samples, random_state=42)
    
    # Add to our list of random samples
    random_samples.append(selected_rows)

# Combine all selected rows into one DataFrame
output_df = pd.concat(random_samples)

# Remove processing columns for CSV output
output_csv_df = output_df.drop(columns=['source_dir', 'video_ext', 'destination_dir'])

# Save to CSV
output_file = '10_per_emotion_experiment/random_10_per_emotion.csv'
output_csv_df.to_csv(output_file, index=False)
print(f"\nSaved {len(output_df)} samples to {output_file}")

# Task 3: Copy corresponding video files for all datasets
total_copied_count = 0
total_not_found_count = 0
all_not_found_files = []

# Process videos from individual dataset samples
for file_output_name, df in dataset_samples.items():
    # Copy video files for this dataset
    copied_count = 0
    not_found_count = 0
    not_found_files = []
    
    for _, row in df.iterrows():
        sample_name = row['name']
        source_dir = row['source_dir']
        video_ext = row['video_ext']
        video_destination_dir = row['destination_dir']
        
        # Ensure the destination directory exists
        os.makedirs(video_destination_dir, exist_ok=True)
        
        video_file = f"{sample_name}{video_ext}"
        source_path = os.path.join(source_dir, video_file)
        destination_path = os.path.join(video_destination_dir, video_file)
        
        if os.path.exists(source_path):
            shutil.copy2(source_path, destination_path)
            copied_count += 1
        else:
            not_found_count += 1
            not_found_files.append(f"{source_dir}/{video_file}")
    
    print(f"\nFor {file_output_name}:")
    print(f"Copied {copied_count} video files to {video_destination_dir}")
    if not_found_count > 0:
        print(f"Could not find {not_found_count} video files")
        print("First 10 missing files:", not_found_files[:10])
        if len(not_found_files) > 10:
            print(f"... and {len(not_found_files) - 10} more")
    
    total_copied_count += copied_count
    total_not_found_count += not_found_count
    all_not_found_files.extend(not_found_files)

print(f"\nTotal summary:")
print(f"Copied {total_copied_count} video files across all datasets")
if total_not_found_count > 0:
    print(f"Could not find {total_not_found_count} video files in total")
