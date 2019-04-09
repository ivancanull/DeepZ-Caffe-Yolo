caffe_root=/home/djn/projects/caffe-gpu
exec_path=${caffe_root}/build/tools/caffe


### train from scratch
#solver_path=${caffe_root}/models/yolo/cfg/yolov3_tiny_solver0.prototxt
#log_path=${caffe_root}/models/yolo/log/yolov3_tiny_train_bn0.out

#nohup \
#  ${exec_path} train \
#  --solver=${solver_path} \
#  --gpu=0 > ${log_path} 2>&1 &



### fine-tune
ft_solver_path=${caffe_root}/models/yolo/cfg/yolov3_tiny_solver0_finetune.prototxt
weight_path=${caffe_root}/models/yolo/weights/yolov3_tiny_train_bn0_iter_300000.caffemodel
ft_log_path=${caffe_root}/models/yolo/log/yolov2_tiny_train_bn0_finetune.out

nohup \
  ${exec_path} train \
  --gpu=0 \
  --solver=${ft_solver_path} \
  --weights=${weight_path} > ${ft_log_path} 2>&1 &
  