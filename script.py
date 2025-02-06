import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--times", type=int, required=True, help="number of times")
parser.add_argument("--file", type=str, default="training/train.py", help="python file")
parser.add_argument("--detector_path", type=str, default="./training/config/detector/fsbi.yaml", help="detector path")
parser.add_argument("--train_dataset", type=str, default="FaceForensics++", help="train dataset")
parser.add_argument("--test_dataset", type=str, default="Celeb-DF-v2", help="test dataset")
args = parser.parse_args()
# CUDA_VISIBLE_DEVICES=3 python script.py --times 10

if __name__ == '__main__':
    for i in range(args.times):
        subprocess.run(['python', args.file,
                        '--detector_path', args.detector_path,
                        '--train_dataset', args.train_dataset,
                        '--test_dataset', args.test_dataset])