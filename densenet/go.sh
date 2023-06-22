# transfer learning.
python -u train.py \
	--data-path /home/ubuntu/dpinsw/classification-resnet/Animal-Detector/datasets/demo3-527classes/images/	\
	--num_classes 527	\
	--weights ./pre-weights/model-9.pth	\
	--batch-size 555	\
	--lr 0.05	\
	--lrf 0.01	\
 	--epochs 10
