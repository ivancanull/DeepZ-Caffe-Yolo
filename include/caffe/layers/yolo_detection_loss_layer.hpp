#ifndef CAFFE_YOLO_DETECTION_LOSS_LAYER_HPP_
#define CAFFE_YOLO_DETECTION_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/box.hpp"
#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes the detection loss for YOLO detector.
 *
 */
template <typename Dtype>
class YoloDetectionLossLayer : public LossLayer<Dtype> {
 public:
  explicit YoloDetectionLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);  

  virtual inline const char* type() const { return "YoloDetectionLoss"; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int ExactNumTopBlobs() const { return 3; }
  
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int IndexEntry(int l, int b, int c, int y, int x) const;
  Box<Dtype> GetPredictionBox(int l, int b, int n, int y, int x, 
      const Dtype* data_array) const;

  void CalcConfidenceLossDiff(int l, int index, Dtype alpha, 
      const Dtype* data_array, Dtype label, Dtype* loss_array, 
      Dtype* diff_array);
  void CalcPositionLossDiff(int l, int index,  int y, int x, int n, Dtype alpha,
      const Dtype* data_array, Box<Dtype> label_box, Dtype* loss_array, 
      Dtype* diff_array);
  void CalcClassLossDiff(int l, int index, Dtype alpha, const Dtype* data_array,
      int label_cls, Dtype* loss_array, Dtype* diff_array);

  void CalcTotalLoss(Dtype* confidence_loss, Dtype* position_loss, Dtype* 
      class_loss);

  int bottom_num_;
  vector<vector<pair<Dtype, Dtype> > > anchor_vec_;
  vector<int> classes_vec_;
  vector<vector<YoloLossType> > loss_type_vec_;
  vector<Dtype> threshold_vec_;
  vector<Dtype> object_scale_vec_;
  vector<Dtype> noobject_scale_vec_;
  vector<Dtype> coord_scale_vec;
  vector<Dtype> class_scale_vec_;
  vector<vector<int> > shape_vec_;
  vector<shared_ptr<Blob<Dtype> > > stage_vec_;
};

}  // namespace caffe

#endif  // CAFFE_YOLO_DETECTION_LOSS_LAYER_HPP_