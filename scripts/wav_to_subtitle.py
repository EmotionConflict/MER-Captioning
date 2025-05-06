import whisper
import torch
import torchaudio
from tqdm.notebook import tqdm
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = whisper.load_model("base").to(DEVICE)

# Set input and output directories
input_dir = "./data/MER_test_subset/test_subset_wav"
output_dir = "./data/MER_test_subset/test_subtitles"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Loop through all .wav files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".wav"):
        file_path = os.path.join(input_dir, filename)
        result = model.transcribe(file_path)
        subtitle_text = result["text"]
        # Save to .txt file with the same base name
        output_filename = os.path.splitext(filename)[0] + ".txt"
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, "w") as f:
            f.write(subtitle_text)
        print(f"Transcribed {filename} -> {output_filename}")