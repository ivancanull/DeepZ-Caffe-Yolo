#include <algorithm>
#include <vector>

#include "caffe/layers/yolo_accuracy_layer.hpp"

namespace caffe {

template <typename Dtype>
inline Dtype sigmoid_act(Dtype x) {
  return Dtype(1) / (Dtype(1) + exp(-x));
}

template <typename Dtype>
void YoloAccuracyLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // clear first
  anchor_vec_.clear();
  classes_vec_.clear();
  shape_vec_.clear();
  // set parameters
  YoloAccuracyParameter acc_param =
      this->layer_param_.yolo_accuracy_param();
  bottom_num_ = acc_param.detection_param_size();
  CHECK_EQ(bottom_num_, bottom.size() - 1);
  for (int l = 0; l < bottom_num_; ++l) {
    // obtain anchors and classes
    YoloDetectionParameter det_param = acc_param.detection_param(l);
    vector<pair<Dtype, Dtype> > anchor_lst;
    for (int n = 0; n < det_param.anchors_size(); ++n) {
      AnchorParameter anchor = det_param.anchors(n);
      anchor_lst.push_back(make_pair(anchor.width(), anchor.height()));
    }
    anchor_vec_.push_back(anchor_lst);
    classes_vec_.push_back(det_param.classes());
    // obtain layer shape
    shape_vec_.push_back(bottom[l]->shape());
  }
}

template <typename Dtype>
void YoloAccuracyLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // set top shape 1 for batch IoU
  vector<int> top_shape(0);
  top[0]->Reshape(top_shape);
  // check
  CHECK_EQ(bottom_num_, bottom.size() - 1);
  CHECK_EQ(bottom.back()->shape(0), bottom[0]->shape(0));
  CHECK_EQ(bottom.back()->count(2), 1);
  // get bottom shape
  for (int l = 0; l < bottom_num_; ++l) {
    // check batch and channel
    CHECK_EQ(bottom[l]->shape(0), bottom[0]->shape(0));
    CHECK_EQ(bottom[l]->shape(1), (5 + classes_vec_[l])*anchor_vec_[l].size());
    // update layer shape
    shape_vec_[l] = bottom[l]->shape();
  }
}

template <typename Dtype>
void YoloAccuracyLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  /// NOTE: currently only support one gt box in one image

  // retrieve ground truth bounding boxes
  const Blob<Dtype>* label_blob = bottom.back();
  const Dtype* label_array = label_blob->cpu_data();
  // <class, coordinates+shape>
  vector<vector<Box<Dtype> > > label_box_vec(label_blob->shape(0));
  for (int b = 0; b < label_blob->shape(0); ++b) {
    for (int i = 0; i < label_blob->shape(1); i += 5) {
      int index = label_blob->shape(1)*b + i;
      if (label_array[index + 0] < Dtype(0))
        break;
      label_box_vec[b].push_back(
          Box<Dtype>(label_array[index + 1], label_array[index + 2], 
                     label_array[index + 3], label_array[index + 4]));
    }
    // make sure only one label box in a single image
    CHECK_EQ(label_box_vec[b].size(), 1);
  }

  // retrieve predictions of each batch and calculate IoU
  int batch_size = shape_vec_[0][0];
  double batch_tot_iou = 0.0;
  for (int b = 0; b < batch_size; ++b) {
    // find the box with highest confidence
    Dtype max_confidence = Dtype(0);
    Box<Dtype> max_box;
    for (int l = 0; l < bottom_num_; ++l) {
      const Dtype* data_array = bottom[l]->cpu_data();
      for (int n = 0; n < anchor_vec_[l].size(); ++n) {
        for (int y = 0; y < shape_vec_[l][2]; ++y) {
          for (int x = 0; x < shape_vec_[l][3]; ++x) {
            int index = IndexEntry(l, b, (5 + classes_vec_[l])*n, y, x);
            Dtype tmp_confidence = sigmoid_act(data_array[index]);
            if (tmp_confidence > max_confidence) {
              max_confidence = tmp_confidence;
              max_box = GetPredictionBox(l, b, n, y, x, data_array);
            }
          }
        }
      }
    }
    
    Dtype min_x = max_box.cx() - max_box.width()  / Dtype(2);
    Dtype min_y = max_box.cy() - max_box.height() / Dtype(2);
    Dtype max_x = max_box.cx() + max_box.width()  / Dtype(2);
    Dtype max_y = max_box.cy() + max_box.height() / Dtype(2);

    // constraints
    if (min_x < Dtype(0)) min_x = Dtype(0);
    if (min_y < Dtype(0)) min_y = Dtype(0);
    if (max_x > Dtype(1)) max_x = Dtype(1);
    if (max_y > Dtype(1)) max_y = Dtype(1);
    
    Dtype cx = (min_x + max_x)/Dtype(2);
    Dtype cy = (min_y + max_y)/Dtype(2);
    Dtype width  = max_x - min_x;
    Dtype height = max_y - min_y;

    // calculate IoU
    Box<Dtype> pred_box(cx, cy, width, height);
    batch_tot_iou += calc_box_iou<Dtype>(label_box_vec[b][0], pred_box);
  }
  // set top
  top[0]->mutable_cpu_data()[0] = batch_tot_iou/batch_size;
}

template <typename Dtype>
int YoloAccuracyLayer<Dtype>::IndexEntry(int l, int b, int c, int y, int x)
    const {
  CHECK_LE(l, bottom_num_);
  return ((b*shape_vec_[l][1] + c)*shape_vec_[l][2] + y)*shape_vec_[l][3] + x;
}

template <typename Dtype>
Box<Dtype> YoloAccuracyLayer<Dtype>::GetPredictionBox(int l, int b, int n, 
    int y, int x, const Dtype* data_array) const {
  int index = IndexEntry(l, b, (5 + classes_vec_[l])*n + 1, y, x);
  int step = shape_vec_[l][2]*shape_vec_[l][3];

  // activate data
  Dtype data0 = sigmoid_act(data_array[index + 0*step]);
  Dtype data1 = sigmoid_act(data_array[index + 1*step]);
  Dtype data2 = data_array[index + 2*step];
  Dtype data3 = data_array[index + 3*step];

  // decode and normalize
  Dtype cx = (x + data0)/shape_vec_[l][3];
  Dtype cy = (y + data1)/shape_vec_[l][2];
  Dtype width  = exp(data2)*anchor_vec_[l][n].first;
  Dtype height = exp(data3)*anchor_vec_[l][n].second;

  return Box<Dtype>(cx, cy, width, height);
}


INSTANTIATE_CLASS(YoloAccuracyLayer);
REGISTER_LAYER_CLASS(YoloAccuracy);

}  // namespace caffe