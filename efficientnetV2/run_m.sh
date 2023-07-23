# Fine tune.
# tensorboard --bind_all --logdir=runs

for name in demo1-331classes  demo2-342classes  demo3-317classes  demo4-106classes  demo5-297classes  demo6-293classes  demo7-323classes  demo8-284classes  demo9-233classes
do
    	echo "Hello, welcome $name"
	echo ../datasets/cls-2nd-infer/$name/
	num_classes="${name:6:3}" 
	echo $num_classes

	./clean.sh
	python train_m.py \
	        --data-path ../datasets/cls-2nd-infer/$name/ \
	        --num_classes $num_classes       \
	        --weights pre-trained/pre_efficientnetv2-m.pth    \
	        --batch-size 42  \
	        --lr 0.005       \
	        --lrf 0.01      \
	        --epochs 6	 \
		> out-run-s.log 2>&1

	save_path=how-to-train/cls-m-$name
	mkdir $save_path
	python convert.py
	mv out-run-s.log models weights onnx runs $save_path
done



