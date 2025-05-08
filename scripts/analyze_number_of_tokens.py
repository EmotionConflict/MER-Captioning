import json
import tiktoken
import matplotlib.pyplot as plt
import numpy as np

# Path to the JSON file
json_path = 'data/MER_test_subset/MER_final_annotations_with_500tokens_peak_frame_desc.json'

# Load the data
with open(json_path, 'r') as f:
    data = json.load(f)

token_counts = []
char_counts = []

# Use OpenAI's tiktoken for GPT-3.5/4 tokenizer
encoding = tiktoken.encoding_for_model('gpt-4o')

for obj in data:
    desc = obj.get('visual_objective_description', '')
    if desc:
        tokens = encoding.encode(desc)
        token_counts.append(len(tokens))
        char_counts.append(len(desc))

# Plot token distribution
plt.figure(figsize=(10, 5))
plt.hist(token_counts, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title('Token Count Distribution for visual_objective_description')
plt.xlabel('Number of Tokens')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Plot character distribution
plt.figure(figsize=(10, 5))
plt.hist(char_counts, bins=30, alpha=0.7, color='green', edgecolor='black')
plt.title('Character Count Distribution for visual_objective_description')
plt.xlabel('Number of Characters')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Print averages
print(f'Average number of tokens: {np.mean(token_counts):.2f}')
print(f'Average number of characters: {np.mean(char_counts):.2f}')
