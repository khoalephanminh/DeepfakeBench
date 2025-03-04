'''
sample script:
> CUDA_VISIBLE_DEVICES=5 python -m analysis.tsne-hoang
'''

import numpy as np
import matplotlib.pyplot as plt
from cuml.manifold import TSNE
#from sklearn.manifold import TSNE
import torch

import os
import sys
import time
import json
from tqdm import tqdm
from datetime import datetime

from training.detectors import DETECTOR
from training.dataset import *


now = datetime.now()
formatted_time = now.strftime("%Y-%m-%d-%H-%M-%S")
output_dir = f'analysis/tsne/{formatted_time}'
os.makedirs(output_dir, exist_ok=True)

config = {
    'output_name': f'{output_dir}/tsne_visualization.png',

    'n_components': 2,
    'perplexity': 30, #Default 30 
    'n_neighbors': 30 * 3,
    'n_iter': 4000,
    'learning_rate': 10,
    'method': 'fft', # 'fft', 'barnes_hut' or 'exact' (default 'fft')
    # 'exact' chay sieu lau, ko nen thu
    'metric': 'consine', # ['l1', 'cityblock', 'manhattan', 'euclidean', 'l2', 'sqeuclidean', 'minkowski', 'chebyshev', 'cosine', 'correlation'] (default 'euclidean')
    'random_state': 1024,

    'model_name': 'fsbi',
    # 'backbone_name': 'efficientnetb4',
    'backbone_config': {
        'mode': 'original',
        'num_classes': 2,
        'inc': 3,
        'dropout': False,
    },
    'backbone_name': 'mixmodel',
    'backbone_config': {
        'model_1':
            'mode': 'original',
            'num_classes': 2,
            'inc': 3,
            'dropout': false,
        'model_2':
            'mode': original,
            'num_classes': 2,
            'inc': 3,
            'dropout': false,
            'pretrained_path': ''./training/pretrained/efficientnet-b4-6ed6700e.pth'.
        'freeze_finetune_path_1': 'logs/training/sbi/sbi_2025-02-06-13-56-37/test/Celeb-DF-v2/ckpt_best.pth',
        num_classes: 2,
    },
    # 'finetune_path': 'logs/training/sbi/sbi_2025-02-06-13-56-37/test/Celeb-DF-v2/ckpt_best.pth',
    'finetune_path': 'logs/training/fsbi/fsbi_2025-02-26-07-24-41/test/Celeb-DF-v2/ckpt_best.pth',
    

    'compression': 'c23',
    'test_batchSize': 32,
    'workers': 8,
    'frame_num': {'test': 32},
    # 'test_dataset': 'Celeb-DF-v2',
    # 'test_dataset': ['FF-F2F', 'FF-DF', 'FF-FS', 'FF-NT'],
    'test_dataset': 'FaceForensics++',
    # 'test_dataset': 'FF-DF',
    'resolution': 380,
    'with_mask': False,
    'with_landmark': False,

    'cuda': True,
    'cudnn': True,
    'loss_func': 'cross_entropy',
    'dataset_json_folder': 'preprocessing/dataset_json',
    # 'label_dict': {'CelebDFv2_real': 0, 'CelebDFv2_fake': 1},
    'label_dict': {'FF-real': 0, 'FF-DF': 1, 'FF-F2F': 2, 'FF-FS': 3, 'FF-NT': 4},
    'rgb_dir': 'datasets/rgb',
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
}

def dump_config():
    with open(f'{output_dir}/config.json', 'w', encoding='utf-8') as file:
        json.dump(config, file, ensure_ascii=False, indent=4)

dump_config()

detector_class = DETECTOR[config['model_name']]
detector = detector_class(config).cuda()

test_set = DeepfakeAbstractBaseDataset(
    config=config,
    mode='test',
)
test_data_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=config['test_batchSize'],
    shuffle=False,
    num_workers=int(config['workers']),
    collate_fn=test_set.collate_fn,
    drop_last = (config['test_dataset']=='DeepFakeDetection'),
)

# Bật chế độ đánh giá (eval) nếu là mô hình deep learning
detector.eval()

# Lưu các feature vectors và labels
features = []
labels = []

# Lặp qua dataset và trích xuất đặc trưng
with torch.no_grad():
    for i, data_dict in tqdm(enumerate(test_data_loader),total=len(test_data_loader)):
        # print(data_dict.keys())
        # > dict_keys(['image', 'label', 'landmark', 'mask'])
        # print(data_dict['label'])
        # > tensor([1])
        
        # get data
        if 'label_spe' in data_dict:
            data_dict.pop('label_spe')  # remove the specific label

        # data_dict['label'] = torch.where(data_dict['label']!=0, 1, 0)  # fix the label to 0 and 1 only

        # move data to GPU elegantly
        for key in data_dict.keys():
            if data_dict[key]!=None:
                data_dict[key]=data_dict[key].cuda()
        # model forward without considering gradient computation
        predictions = detector(data_dict)
        labels += list(data_dict['label'].cpu().numpy())
        features += list(predictions['feat'].cpu().numpy())

# Free detector
del detector
torch.cuda.empty_cache()

# Chuyển list sang numpy array
features = np.array(features)
print(f"Feature shape: {features.shape}")

# Giảm chiều với t-SNE
start_time = time.time()
tsne = TSNE(n_components=config['n_components'],
            perplexity=config['perplexity'],
            n_neighbors=config['n_neighbors'],
            random_state=config['random_state'],
            n_iter=config['n_iter'],
            learning_rate=config['learning_rate'],
            metric=config['metric'])
X_embedded = tsne.fit_transform(features)
end_time = time.time()

# Vẽ scatter plot và lưu vào file
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap="jet", alpha=0.7)
plt.colorbar(scatter, label="Labels")
plt.title("t-SNE Visualization of Model Features")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")

# Lưu file thay vì hiển thị
plt.savefig(config['output_name'], dpi=300)  # Lưu với độ phân giải cao
plt.close()  # Đóng figure để tránh chiếm bộ nhớ

print(f"t-SNE visualization completed in {end_time - start_time:.2f} seconds, stored at {config['output_name']}")
config['total_time'] = end_time - start_time
dump_config()