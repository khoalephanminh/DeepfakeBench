import os


txt_path = './backup_paths.txt'

sum_size = 0
with open(txt_path, 'r') as f:
    # for line without '\n'
    for line in f:
        line = line.strip()
        full_path_log = f'/raid/dtle/deepfake/DeepfakeBench/{line}'
        ckpt = line.replace('training.log', 'test/Celeb-DF-v2/ckpt_best.pth')
        full_path_ckpt = f'/raid/dtle/deepfake/DeepfakeBench/{ckpt}'
        print(ckpt)
        #calculate sum of Mb
        size = os.path.getsize(full_path_ckpt)
        sum_size += size

        save_path_ckpt = f'/raid/dtle/deepfake/DeepfakeBench/log_backup/{ckpt}'
        save_path_log = f'/raid/dtle/deepfake/DeepfakeBench/log_backup/{line}'
        os.makedirs(os.path.dirname(save_path_ckpt), exist_ok=True)
        os.makedirs(os.path.dirname(save_path_log), exist_ok=True)
        # print("save_path_ckpt=", save_path_ckpt)
        # print("save_path_log=", save_path_log)
        os.system(f"cp {full_path_ckpt} {save_path_ckpt}")
        os.system(f"cp {full_path_log} {save_path_log}")

print("sum_size=", sum_size/1024/1024, "Mb")
#
