# NOTE: This one works. 

import os
import subprocess
import argparse
import pandas as pd
import glob

# AU to facial phrase mapping
AU_PHRASES = {
    'AU01': 'Inner Brow Raiser',
    'AU02': 'Outer Brow Raiser',
    'AU04': 'Brow Lowerer',
    'AU05': 'Upper Lid Raiser',
    'AU06': 'Cheek Raiser',
    'AU07': 'Lid Tightener',
    'AU09': 'Nose Wrinkler',
    'AU10': 'Upper Lip Raiser',
    'AU12': 'Lip Corner Puller',
    'AU14': 'Dimpler',
    'AU15': 'Lip Corner Depressor',
    'AU17': 'Chin Raiser',
    'AU20': 'Lip stretcher',
    'AU23': 'Lip Tightener',
    'AU25': 'Lips Part',
    'AU26': 'Jaw Drop',
    'AU28': 'Lip Suck',
    'AU45': 'Blink'
}

# AU Intensity Mapping
def map_au_intensity(value):
    if value < 0.2:
        return "barely"
    elif value < 1.0:
        return "slightly"
    elif value < 2.5:
        return "moderately"
    elif value < 5.0:
        return "strongly"
    else:
        return "very strongly"

def extract_au_from_video(video_path, output_dir, openface_bin_path):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Construct the command to run OpenFace's FeatureExtraction
    command = [
        openface_bin_path,
        "-f", video_path,
        "-aus",                      # Enable AU detection
        "-out_dir", output_dir       # Where to save the .csv output
    ]

    print(f"Processing: {os.path.basename(video_path)}")
    print(f"Running command:\n{' '.join(command)}\n")
    subprocess.run(command, check=True)
    print(f"AU data saved to: {output_dir}")

def find_peak_frame(au_data_path):
    # === LOAD AU DATA ===
    df = pd.read_csv(au_data_path)

    # Extract AU presence and intensity columns
    au_presence_cols = [col for col in df.columns if "_c" in col]
    au_intensity_cols = [col for col in df.columns if "_r" in col]

    # Step 1: Identify most frequently occurring AUs
    au_frequencies = df[au_presence_cols].sum().sort_values(ascending=False)
    most_frequent_aus = au_frequencies.head(3).index.str.replace("_c", "")

    # Step 2: Sum the intensity values of these AUs for each frame
    relevant_intensity_cols = [au + "_r" for au in most_frequent_aus]
    df["emotion_sum"] = df[relevant_intensity_cols].sum(axis=1)

    # Step 3: Find emotional peak frame
    peak_frame_index = df["emotion_sum"].idxmax()
    return peak_frame_index, peak_frame_index/30


def parse_au_intensity(openface_csv_path, peak_index):
    try:
        df = pd.read_csv(openface_csv_path)
        if peak_index >= len(df):
            peak_index = len(df) // 2  # fallback to middle if peak invalid
        row = df.iloc[peak_index]

        au_phrases = []
        peak_aus = {}

        for au in AU_PHRASES.keys():
            if f"{au}_r" in row:
                value = row[f"{au}_r"]
                if value > 0.1:
                    intensity = map_au_intensity(value)
                    phrase = AU_PHRASES[au]
                    full_phrase = f"{intensity} {phrase}"
                    au_phrases.append(full_phrase)
                peak_aus[au]=value

        return au_phrases, peak_aus
    except Exception as e:
        print(f"[ERROR] AU parsing failed for {openface_csv_path}: {e}")
        return [], []

def process_video_files(video_files, output_dir, openface_bin_path):
    """Process multiple video files and extract AU data"""
    results = {}
    
    for video_path in video_files:
        try:
            # Get video filename without extension
            video_name = os.path.basename(video_path).split('.')[0]
            
            # Extract AUs from video
            extract_au_from_video(video_path, output_dir, openface_bin_path)
            
            # Find peak frame in the extracted data
            csv_path = os.path.join(output_dir, f"{video_name}.csv")
            peak_index, peak_time = find_peak_frame(csv_path)
            
            # Parse AU intensities at peak frame
            au_phrases, peak_aus = parse_au_intensity(csv_path, peak_index)
            
            # Store results
            results[video_name] = {
                'peak_frame': peak_index,
                'peak_time': peak_time,
                'au_phrases': au_phrases,
                'au_data': peak_aus
            }
            
            print(f"Processed {video_name}:")
            print(f"  Peak frame: {peak_index} (at {peak_time:.2f} seconds)")
            print(f"  AU phrases: {', '.join(au_phrases)}")
            print("-----------------------------------")
            
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Action Units from videos using OpenFace")
    parser.add_argument("--data_dir", type=str, default="data/MER_test_subset/test_subset",
                       help="Directory containing the video files")
    parser.add_argument("--output_dir", type=str, default="AU_labeling/output",
                       help="Directory to save OpenFace outputs")
    parser.add_argument("--openface_bin", type=str, 
                       default="/Users/charlie/LLMs/OpenFace/build/bin/FeatureExtraction",
                       help="Path to OpenFace FeatureExtraction binary")
    parser.add_argument("--limit", type=int, 
                       help="Optional: Limit the number of files to process")
    parser.add_argument("--specific_files", nargs='+', 
                       help="Specific files to process (overrides other options)")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.specific_files:
        # Process specific files provided by user
        video_files = [os.path.join(args.data_dir, f) for f in args.specific_files]
    else:
        # Get all MP4 files in the data directory
        all_videos = glob.glob(os.path.join(args.data_dir, "*.mp4"))
        
        # Apply limit if specified
        if args.limit:
            video_files = all_videos[:args.limit]
        else:
            video_files = all_videos  # Process all files
    
    # Print information about processing
    print(f"Found {len(video_files)} video files to process")
    
    # Process the selected video files
    results = process_video_files(video_files, args.output_dir, args.openface_bin)
    
    # Print summary
    print("\nSummary of processed videos:")
    for video_name, data in results.items():
        print(f"{video_name}: {len(data['au_phrases'])} AUs detected at peak frame {data['peak_frame']}")
    
    print(f"\nTotal videos processed: {len(results)}")
