# how to set up

1. git clone git@github.com:QwenLM/Qwen-Audio.git
2. `cd Qwen-Audio`
3. `pip install -r requirements.txt`

# MER-Captioning

## How to extract - Audio Tone Description

Output: "The woman in the video speaks with an excited voice."
Model: Qwen-Audio

# data and folder structures

For the short experiment 10 videos / emotion, I am using the mer2023test1&2/ and test-labels/ folders. Please contact Chenyu Zhang if you'd like to obtain the these folders.

# stats when you run sample_10_per_emotion.py

NOTE: it creates a new folder called `10_per_emotion_experiment`.
For the MER2023 test dataset, test1 is .avi files and test2 contains .mp4 files.
The scripts sample for both .avi and .mp4 and later I've manually moved the .mp4 files to the data folder to follow the structure similar to MELD.

Processing mer2023-test-labels/test1-label.csv...
Emotions in test1-label.csv:
worried: 57
happy: 93
neutral: 88
sad: 50
angry: 105
surprise: 18
Total unique emotions in this set: 6
Saved 60 samples from test1-label.csv to 10_per_emotion_experiment/random_10_per_emotion_test1.csv

Processing mer2023-test-labels/test2-label.csv...
Emotions in test2-label.csv:
happy: 87
worried: 52
angry: 98
neutral: 123
sad: 40
surprise: 12
Total unique emotions in this set: 6
Saved 60 samples from test2-label.csv to 10_per_emotion_experiment/random_10_per_emotion_test2.csv

Combined emotions across all datasets:
worried: 109
happy: 180
neutral: 211
sad: 90
angry: 203
surprise: 30
Total unique emotions across all datasets: 6

Saved 60 samples to 10_per_emotion_experiment/random_10_per_emotion.csv

For random_10_per_emotion_test1.csv:
Copied 60 video files to 10_per_emotion_experiment/videos-avi

For random_10_per_emotion_test2.csv:
Copied 60 video files to 10_per_emotion_experiment/videos-mp4

Total summary:
Copied 120 video files across all datasets

# how to set up openface manually

https://pranav-srivastava.medium.com/openface-2-0-mac-installation-and-pose-detection-257289cbc79b

# extract AUs

`python scripts/au_extraction.py`

# Using GPT-40 to extract the data. BLIP-2 doesn't have a good quality

`KMP_DUPLICATE_LIB_OK=TRUE python scripts/peak_frame_description.py`

# Colab code to process the data

https://colab.research.google.com/drive/16G0X94hrFYtICsbwSECzTISL7sFJ2aZ1#scrollTo=BMtiHza5p7mN
