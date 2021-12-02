python train.py \
    --data_dir /vision/u/naagnes/video/train \
    --accelerator gpu \
    --gpus -1 \
    --strategy ddp \
    --num_sanity_val_steps 0 \
    --max_epochs 100 \
    --limit_train_batches 0.002 \
    --batch_size 12 \
    --num_workers 6 \
    --profiler simple 

