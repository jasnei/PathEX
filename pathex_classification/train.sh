#!/bin/bash
# Directory
TRAIN_DIR=''
SAVE_DIR="./checkpoints/experiment/patch_512"
TEST_DIR="/data/experiment/patch_512/test"

# Training parameters
IMAGE_SIZE=512
EPOCHS=100
MIXUPOFFEPOCH=60
BS=64
NWORKERS=2
NGPU=2
RESUME=''
DINO=''

echo "image_size: $IMAGE_SIZE, epoch: $EPOCHS, ngpu: $NGPU, batch size: $BS, num workers: $NWORKERS"

# List directory
filenames=`ls $TRAIN_DIR`

for filename in $filenames; do
    # Concat dir and filename
    filepath="$TRAIN_DIR/$filename"
    savepath="${SAVE_DIR}_${filename}"
    if test -d $filepath
    then
        # echo $filename
        echo "data directory: $filepath"
        echo "save directory: $savepath"

        # nohup \
        torchrun --standalone --nnodes=1 --nproc_per_node=$NGPU \
                train.py \
                        --data-dir=$filepath \
                        --seed=42 \
                        --train-ratio=0.8 \
                        --test-dir=$TEST_DIR \
                        --batch-size=$BS \
                        --valid-batch-size=$BS \
                        --pin_memory=False \
                        --num_workers=$NWORKERS \
                        --drop_last=False \
                        --save-dir=$savepath \
                        --log-interval=100 \
                        --image-size=$IMAGE_SIZE \
                        --pretrained=True \
                        --resume=$RESUME \
                        --weight=None \
                        --epochs=$EPOCHS \
                        --lr=0.0001 \
                        --warmup-epochs=5 \
                        --warmup-lr=0.000001 \
                        --mixup=0.15 \
                        --cutmix=0.2 \
                        --smoothing=0.05 \
                        --mixup_off_epoch=$MIXUPOFFEPOCH \
                        --amp=True \
                        --dino_path=$DINO \
    fi
done

# RESUME='/opt/data/default/albert/projects/breast_cancer_classification/checkpoints/V1.2/R50_1024_v1.2_dino_20230607-100128/model_66_0.9910.pth.tar'
# DINO='/opt/data/default/albert/projects/breast_unk/1_pretrain/checkpoints/r50/checkpoint_convnext.pth'


