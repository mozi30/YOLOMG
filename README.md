# YOLOMG
Codes and dataset for the paper "YOLOMG: Extremely Small Drone-to-Drone Detection with Appearance and Pixel-level Motion Fusion"

# Dataset
ARD100 dataset
- [BaiduYun](https://pan.baidu.com/s/1ycAoKbzQ1rlzvKr8VRakgw?pwd=1x2z ) (code:1x2z)

# data processing
## dataset spliting
python3 split_train_val.py --xml_path xx/xxx/Annotations --txt_path xx/xxx/ImageSets/Main
## transfer voc label to yolo label with .txt files
python3 voc2yolo.p
## generate images directory, train.txt, val.txt, test.txt
python3 voc_label.py
## generate mask directory, train2.txt, val2.txt, test2.txt
Python3 voc_label2.py

# train
python3 train.py --data data/NPS.yaml --cfg models/dual_uav2.yaml --weights yolov5s.pt --batch-size 8 --epochs 100 --imgsz 1280 --name NPS-1280

# val
python3 val.py --weights runs/train/NPS-1280/weights/best.pt --data data/NPS_test.yaml --task val --conf-thres 0.001 --name NPS_test-1280 --imgsz 1280 --batch-size 8 --device 0

# DDP train
python -m torch.distributed.run --nproc_per_node=4 --master_port 12345 train.py --data data/ARD100_mask32.yaml --cfg models/ARD100_drone_s.yaml --weights yolov5s.pt --batch-size 16 --epochs 100 --imgsz 1280 --name ARD100_mask32-1280 --device 0,1,2,3
