caffe_root=/home/djn/projects/caffe-gpu
exec_path=${caffe_root}/build/tools/caffe
solver_path=${caffe_root}/models/yolo/cfg/yolov2_tiny_solver1.prototxt
log_path=${caffe_root}/models/yolo/log/yolov2_tiny_train_bn1.out

#nohup \
  ${exec_path} train \
  --solver=${solver_path} \
  --gpu=0 #> ${log_path} 2>&1 &