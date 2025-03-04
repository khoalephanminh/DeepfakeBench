'''
sample script:
> CUDA_VISIBLE_DEVICES=5 python -m analysis.tsne \
    --config_path analysis/configs/tsne-hoang.yaml
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
import yaml
import argparse
from tqdm import tqdm
from datetime import datetime

from training.detectors import DETECTOR
from training.dataset import *
from torch.nn.parallel import DistributedDataParallel as DDP


parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--config_path', type=str,
                    default='/data/home/zhiyuanyan/DeepfakeBenchv2/training/config/detector/sbi.yaml',
                    help='path to detector YAML file')
args = parser.parse_args()


now = datetime.now()
formatted_time = now.strftime("%Y-%m-%d-%H-%M-%S")
output_dir = f'analysis/tsne/{formatted_time}'

def dump_config(config):
    os.makedirs(output_dir, exist_ok=True)
    with open(f'{output_dir}/config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=None)

def main():
    ########### parse options and load config ###########
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['output_dir'] = f'{output_dir}'
    dump_config(config)

    ########### model ###########
    detector_class = DETECTOR[config['model_name']]
    detector = detector_class(config).cuda()
    # load ckpt
    if os.path.isfile(config['finetune_path']):
        saved = torch.load(config['finetune_path'], map_location='cpu')
        suffix = config['finetune_path'].split('.')[-1]
        if suffix == 'p':
            saved = saved.state_dict()
        if type(detector) is DDP:
            # add the prefix 'module.' to the keys if needed
            saved = {k.replace('backbone.', 'module.backbone.'): v for k, v in saved.items()}
            saved = {k.replace('module.module.', 'module.'): v for k, v in saved.items()}
        else:
            # remove the prefix 'module.' from the keys if needed
            saved = {k.replace('module.backbone.', 'backbone.'): v for k, v in saved.items()}
        detector.load_state_dict(saved)
    else:
        raise NotImplementedError(
            "=> no model found at '{}'".format(config['finetune_path']))

    ########### dataset ###########
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

    ########### t-SNE ###########

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
    for perplexity in config['perplexities']:
        perplexity = config['perplexity']
        output_name = f"{output_dir}/tSNE_perplexity_{perplexity}.png"
        tsne = TSNE(n_components=config['n_components'],
                    perplexity=perplexity,
                    n_neighbors=config['n_neighbors'],
                    random_state=config['random_state'],
                    n_iter=config['n_iter'],
                    learning_rate=config['learning_rate'],
                    metric=config['metric'])
        X_embedded = tsne.fit_transform(features)

        # Vẽ scatter plot và lưu vào file
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap="jet", alpha=0.7)
        plt.colorbar(scatter, label="Labels")
        plt.title(f"t-SNE Visualization with Perplexity={perplexity}")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")

        # Lưu file thay vì hiển thị
        plt.savefig(output_name, dpi=300)  # Lưu với độ phân giải cao
        plt.close()  # Đóng figure để tránh chiếm bộ nhớ
    end_time = time.time()

    print(f"t-SNE visualization completed in {end_time - start_time:.2f} seconds, stored at {config['output_dir']}")
    config['total_time'] = end_time - start_time
    dump_config(config)


if __name__ == "__main__":
    main()