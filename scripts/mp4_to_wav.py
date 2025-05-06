# pip install moviepy

import os
from moviepy.editor import VideoFileClip

input_dir = 'data/MER_test_subset/test_subset'
output_dir = 'data/MER_test_subset/test_subset_wav'

os.makedirs(output_dir, exist_ok=True)

def convert_mp4_to_wav(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith('.mp4'):
            mp4_path = os.path.join(input_dir, filename)
            wav_filename = os.path.splitext(filename)[0] + '.wav'
            wav_path = os.path.join(output_dir, wav_filename)
            print(f'Converting {mp4_path} to {wav_path}')
            with VideoFileClip(mp4_path) as video:
                audio = video.audio
                if audio is not None:
                    audio.write_audiofile(wav_path, codec='pcm_s16le')
                else:
                    print(f'No audio track found in {mp4_path}')

if __name__ == '__main__':
    convert_mp4_to_wav(input_dir, output_dir)
