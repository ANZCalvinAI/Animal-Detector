# Fine tune.
nohup python -u train.py \
	--data-path /home/ubuntu/dpinsw/classification-resnet/Animal-Detector/datasets/demo3-527classes/images/	\
	--num_classes 527	\
	--weights ./pre-weights/model-9.pth	\
	--batch-size 200	\
	--lr 0.05	\
	--lrf 0.01	\
 	--epochs 10	\
 	> out-run1.log 2>&1 &
