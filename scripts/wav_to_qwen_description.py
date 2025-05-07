# ! pip install transformers_stream_generator
# ! pip install tiktoken

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import os

# Set the following seed to reproduce results if needed
# torch.manual_seed(1234)

# Use CPU if CUDA is not available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-Audio-Chat", trust_remote_code=True)

# Load model to the appropriate device
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio-Chat", trust_remote_code=True).to(device).eval()

# Load generation configuration
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-Audio-Chat", trust_remote_code=True)

# Define input and output directories
input_dir = "./data/MER_test_subset/test_subset_wav"
output_dir = "./data/MER_test_subset/test_subset_gwen_description"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# List all .wav files in the input directory
wav_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]

for wav_file in wav_files:
    wav_path = os.path.join(input_dir, wav_file)
    # Format the query for each file
    query = tokenizer.from_list_format([
        {"audio": wav_path},
        {"text": "Describe the speaker’s vocal delivery in this audio. Focus only on how they sound, not what they are saying. Include detailed observations about: \n- Prosody: pitch, intonation, pacing, loudness, rhythm, and pauses or hesitation \n- Voice quality: breathiness, tension, harshness, smoothness, or creakiness \n- Articulation: clarity, enunciation, and any slurring, stuttering, or irregular speech patterns \nAlso describe how expressive or monotone the speaker sounds. Be as descriptive as possible and explain what these vocal features might suggest about their emotional tone, without interpreting the actual words."}
    ])
    # Generate the response
    response, history = model.chat(tokenizer, query=query, history=None)
    # Save the response to a .txt file with the same base name
    output_file = os.path.splitext(wav_file)[0] + ".txt"
    output_path = os.path.join(output_dir, output_file)
    with open(output_path, 'w') as f:
        f.write(response)

print(f"Processed {len(wav_files)} files. Descriptions saved to {output_dir}.")


