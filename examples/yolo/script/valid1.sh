caffe_root=/home/djn/projects/caffe-gpu
exec_path=${caffe_root}/build/tools/caffe
model_path=${caffe_root}/models/yolo/cfg/yolov2_tiny_valid_bn1.prototxt
weights_path=${caffe_root}/models/yolo/weights/yolov2_tiny_train_bn1_iter_180000.caffemodel

#nohup \
${exec_path} test \
  --gpu=1 \
  --model=${model_path} \
  --weights=${weights_path} \
  --iterations=1960 #> ${log_path} 2>&1 &