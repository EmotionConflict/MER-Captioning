import os
import json
import csv

# Input and output file paths
input_path = 'data/MER_test_subset/first_step.json'
output_path = 'data/MER_test_subset/MER_final_annotations.json'
audio_desc_dir = 'data/MER_test_subset/test_subset_gwen_description'
visual_obj_desc_dir = 'data/MER_test_subset/test_subset_peak_frame_description'
caption_dir = 'data/MER_test_subset/test_subtitles'

def get_audio_description(sample_id):
    txt_file = os.path.join(audio_desc_dir, f'{sample_id}.txt')
    if os.path.exists(txt_file):
        with open(txt_file, 'r') as f:
            return f.read().strip()
    return ''

def get_visual_objective_description(sample_id):
    csv_file = os.path.join(visual_obj_desc_dir, f'{sample_id}.csv')
    if os.path.exists(csv_file):
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                return row.get('description', '').strip()
    return ''

def get_caption(sample_id):
    txt_file = os.path.join(caption_dir, f'{sample_id}.txt')
    if os.path.exists(txt_file):
        with open(txt_file, 'r') as f:
            return f.read().strip()
    return ''

def convert_first_step_to_final_annotations(input_path, output_path):
    with open(input_path, 'r') as infile:
        data = json.load(infile)

    output = []
    for sample_id, sample in data.items():
        entry = {
            'video_id': f'{sample_id}.mp4',
            'peak_time': sample.get('peak_time', 0.0),
            'visual_expression_description': sample.get('au_phrases', []),
            'visual_objective_description': get_visual_objective_description(sample_id),
            'raw_AU_values_at_peak': sample.get('au_data', {}),
            'coarse-grained_summary': '',
            'fine-grained_summary': '',
            'audio_description': get_audio_description(sample_id),
            'caption': get_caption(sample_id)
        }
        output.append(entry)

    with open(output_path, 'w') as outfile:
        json.dump(output, outfile, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    convert_first_step_to_final_annotations(input_path, output_path)
