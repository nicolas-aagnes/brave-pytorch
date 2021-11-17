# Script arguments: scratch space folder, eg /scr or /scr-ssd

DATA_FOLDER=$1/naagnes/kin400
echo "Using $DATA_FOLDER as dataset folder."

mkdir -p $DATA_FOLDER

if [ ! -d "$DATA_FOLDER/video" ]
then
    echo "Syncing contents in /vision/group/Kinetics/kin400/video.tar.gz to $DATA_FOLDER"
    cp /vision/group/Kinetics/kin400/video.tar.gz $DATA_FOLDER

    echo "Unzipping contents in $DATA_FOLDER/video.tar.gz"
    tar -xf $DATA_FOLDER/video.tar.gz -C $DATA_FOLDER
    rm -rf $DATA_FOLDER/video.tar.gz
else
    echo "Folder $DATA_FOLDER/video is already present. Skipping copying."
fi

## TRAIN
echo "Using python from $(which python)"

python train.py --accelerator gpu --gpus 1 --max_steps 10 --batch_size 4