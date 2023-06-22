# Fine tune.
nohup python -u train.py \
	--data-path /home/ubuntu/dpinsw/classification-resnet/Animal-Detector/datasets/demo3-527classes/images/	\
	--num_classes 527	\
	--weights /home/ubuntu/dpinsw/classification-resnet/deep-learning-for-image-processing/pytorch_classification/Test8_densenet/pretrained/densenet161-8d451a50.pth	\
	--batch-size 80	\
	--lr 0.05	\
	--lrf 0.01	\
 	--epochs 10	\
 	> out-run1.log 2>&1 &
