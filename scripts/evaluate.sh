python evaluate.py \
    --data_dir /vision/group/UCF-101 \
    --annotation_path /vision/group/ucf101/ucfTrainTestlist \
    --accelerator gpu \
    --gpus 1 \
    --batch_size 24 \
    --num_workers 4
    