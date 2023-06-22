# Fine tune.
# tensorboard --bind_all --logdir=runs

python -u train.py \
        --data-path /home/ubuntu/dpinsw/classification-resnet/Animal-Detector/datasets/demo3-527classes/images/ \
        --num_classes 527       \
        --weights ./pre-weights/model-9.pth     \
        --batch-size 200 \
        --lr 0.1       \
        --lrf 0.001      \
        --epochs 8

