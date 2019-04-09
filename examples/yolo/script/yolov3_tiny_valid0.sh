caffe_root=/home/djn/projects/caffe-gpu
exec_path=${caffe_root}/build/tools/caffe
model_path=${caffe_root}/models/yolo/cfg/yolov3_tiny_valid_bn0.prototxt
weights_path=${caffe_root}/models/yolo/weights/yolov3_tiny_train_bn0_ft_iter_30000.caffemodel

#nohup \
${exec_path} test \
  --gpu=0 \
  --model=${model_path} \
  --weights=${weights_path} \
  --iterations=1966 #> ${log_path} 2>&1 &