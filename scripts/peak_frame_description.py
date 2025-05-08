# NOTE: used this file to extract the data in the test_subset_peak_frame_description folder

import cv2
from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import pandas as pd
import os
import csv
import openai
import base64
import io
import warnings
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()  # Uses api_key from env

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

# --- Step 1: Extract specific frame ---
def extract_frame_by_index(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Could not read frame at index {frame_idx}.")
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)

# --- Step 2: Load BLIP-2 model ---
def load_blip2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xl",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ).to(device)
    return processor, model, device

def describe_image_with_openai(image, prompt):
    """
    Describe an image using OpenAI's GPT-4 Vision API.
    Args:
        image (PIL.Image): The image to describe.
        prompt (str): The prompt to send to the model.
    Returns:
        str: The generated description.
    """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                ]
            }
        ],
        max_tokens=300,
    )
    return response.choices[0].message.content

# --- Config ---
video_dir = "./data/MER_test_subset/test_subset"
csv_dir = "./data/MER_test_subset/test_subset_au"
out_dir = "./data/MER_test_subset/openai_test_subset_peak_frame_description"
prompt = "Describe what is happening in this video frame as if you're narrating it to someone who cannot see it. Focus only on visible details such as people's actions, facial expressions, gestures, body language, clothing, objects, and the physical setting. Be specific about how people are positioned and how they interact with each other and their surroundings. Write descriptively—do not simply list objects. Include visual cues that might suggest emotional states, but don't speculate beyond what's visible."

os.makedirs(out_dir, exist_ok=True)

video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

# Only process the first 2 videos for testing
# video_files = video_files[:2]

# --- Parameter to control which model to use ---
use_openai = True  # Set to False to use BLIP-2

# Load BLIP-2 only if needed
if not use_openai:
    try:
        processor, model, device = load_blip2()
    except Exception as e:
        warnings.warn(f"BLIP-2 model could not be loaded: {e}")
        processor = model = device = None

for video_file in video_files:
    base_name = os.path.splitext(video_file)[0]
    video_path = os.path.join(video_dir, video_file)
    csv_path = os.path.join(csv_dir, base_name + ".csv")
    out_path = os.path.join(out_dir, base_name + ".csv")

    if not os.path.exists(csv_path):
        print(f"CSV not found for {video_file}, skipping.")
        continue

    try:
        frame_index, time = find_peak_frame(csv_path)
        image = extract_frame_by_index(video_path, frame_index)
        if use_openai:
            generated_text = describe_image_with_openai(image, prompt)
        else:
            if processor is None or model is None or device is None:
                raise RuntimeError("BLIP-2 model is not available.")
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16 if torch.cuda.is_available() else torch.float32)
            generated_ids = model.generate(**inputs)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # Save result to CSV
        with open(out_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["peak_frame_index", "description"])
            writer.writerow([frame_index, generated_text])
        print(f"Saved: {out_path}")
    except Exception as e:
        print(f"Error processing {video_file}: {e}")