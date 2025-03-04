import os
import json

dataset_folder = '/raid/dtle/deepfake/DeepfakeBench/preprocessing/dataset_json/Celeb-DF-v2.json'
dataset_name = dataset_folder.split('/')[-1].split('.')[0]
print("dataset_name: ", dataset_name)

# Load JSON data
with open(dataset_folder, 'r') as f:
    data = json.load(f)

# Function to modify frame paths
def modify_frame_paths(data):
    for key1, value1 in data.items():
        for key2, value2 in value1.items():
            for key3, value3 in value2.items():
                for key4, value4 in value3.items():
                    if 'frames' in value4:
                        value4['frames'] = [frame.replace(dataset_name, f'{dataset_name}-cropped0125') for frame in value4['frames']]

# Modify frame paths
modify_frame_paths(data)

# Wrap everything in a new dictionary with the new key
new_data = {f'{dataset_name}-cropped0125': data[dataset_name]}

# Save updated JSON data
save_path = dataset_folder.replace(dataset_name, f'{dataset_name}-cropped0125')
with open(save_path, 'w') as f:
    json.dump(new_data, f, indent=4)