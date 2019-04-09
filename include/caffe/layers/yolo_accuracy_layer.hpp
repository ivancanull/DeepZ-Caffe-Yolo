#ifndef CAFFE_YOLO_ACCURACY_LAYER_HPP_
#define CAFFE_YOLO_ACCURACY_LAYER_HPP_

#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/box.hpp"

namespace caffe {

template <typename Dtype>
class YoloAccuracyLayer : public Layer<Dtype> {
 public:
  explicit YoloAccuracyLayer(const LayerParameter& param)  
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "YoloAccuracy"; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }

  int IndexEntry(int l, int b, int c, int y, int x) const;
  Box<Dtype> GetPredictionBox(int l, int b, int n, int y, int x, 
      const Dtype* data_array) const;

  int bottom_num_;
  vector<vector<pair<Dtype, Dtype> > > anchor_vec_;
  vector<int> classes_vec_;
  vector<vector<int> > shape_vec_;
};

}  // namespace caffe


#endif  // CAFFE_YOLO_ACCURACY_LAYER_HPP_