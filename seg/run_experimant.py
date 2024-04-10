import pathlib
import subprocess
import datetime
import os
import argparse

# tests = ['A', 'B','C','D','E']
tests = range(5)
# tests = [0,1]

##
# 
# #

parser = argparse.ArgumentParser()
parser.add_argument("--prj", type=pathlib.Path, required=True)
parser.add_argument("--exp", type=str, required=True) # 実験名
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--pred_only", action="store_true") #推論のみ行う場合

args = parser.parse_args()

yolov7_dir = pathlib.Path("./")
src_dir = pathlib.Path("./src")

project_dir = args.prj
experiment_dir = args.prj / "experiments" / args.exp
datasets_dir = args.prj / "dataset"

batch_size = args.batch
epochs = args.epochs
# pre_trained_weight = '""'
pre_trained_weight = 'yolov7-seg.pt'

img_size = 512

# TODO: current_datdaset ディレクトリに画像ファイルのパスのテキストファイルをコピーする

pred_only = args.pred_only

for test in tests:
    fold = f"fold{test}"
    
    dataset_dir = datasets_dir / fold
    experiment_fold_dir = experiment_dir / fold
    os.makedirs(experiment_fold_dir, exist_ok=True)


    if not (pred_only):
        print(f"Train {fold}")
        dt_train_start = datetime.datetime.now()

        train_command = f"cd {yolov7_dir} && python segment/train.py \
            --weights {pre_trained_weight} \
            --cfg models/segment/yolov7-seg.yaml \
            --data {dataset_dir}/config.yaml \
            --epochs {epochs} \
            --batch-size {batch_size} \
            --img-size {img_size} \
            --workers 8 \
            --project {experiment_dir} \
            --name {fold} \
            --exist-ok \
            --device 0"
        print(train_command.replace("   ", ""))
        subprocess.run(train_command, shell=True)

        dt_train_end = datetime.datetime.now()
        td_train = dt_train_end - dt_train_start

    # 推論
    dt_pred_start = datetime.datetime.now()

    prediction_command = f"python3 segment/predict.py \
        --weights {experiment_fold_dir /'weights' / 'last.pt'} \
        --source {dataset_dir /'test'/ 'images'} \
        --project {experiment_dir} \
        --conf-thres 0.1 \
        --exist-ok \
        --name predictions \
        --save-txt \
        --save-conf \
        --device 0"
    print(f"{fold} : {prediction_command.replace('  ','')}")
    subprocess.run(prediction_command, shell=True)

    dt_pred_end = datetime.datetime.now()
    td_pred = dt_pred_end - dt_pred_start

    if not (pred_only):

        with open(experiment_fold_dir / "train_report.txt", "w") as f:
            f.write(f"Batch size: {batch_size}\n")
            f.write(f"Epochs: {epochs}\n")

            f.write("== TRAIN ==\n")
            f.write(f"Start: {dt_train_start}\n")
            f.write(f"End: {dt_train_end}\n")
            f.write(f"Duration: {td_train}\n\n")
            f.write("== PREDICTION ==\n")
            f.write(f"Start: {dt_pred_start}\n")
            f.write(f"End: {dt_pred_end}\n")
            f.write(f"Duration: {td_pred}\n")
