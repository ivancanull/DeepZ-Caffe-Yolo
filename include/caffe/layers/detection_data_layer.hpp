#ifndef CAFFE_DETECTION_DATA_LAYER_HPP_
#define CAFFE_DETECTION_DATA_LAYER_HPP_

#include <string>
#include <vector>

#include "caffe/blob.hpp"
//#include "caffe/data_transformer.hpp"
#include "caffe/common.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "boost/random.hpp"

namespace det_rng {

class FloatRNG {
 public:
  FloatRNG() {}
  void InitRand() { this->rng.seed(static_cast<unsigned int>(std::time(0))); }
  void InitRand(unsigned int seed) { this->rng.seed(seed); }
  float Rand(float min, float max) {
    if(max < min){
        float tmp = min;
        min = max;
        max = tmp;
    }
    else if (max == min) {  // A BUG occurs if don't return earlily
      return max;
    }
    boost::uniform_real<float> uni_dist(min, max);
    boost::variate_generator<boost::mt19937&, boost::uniform_real<float> >
        variate_generator(this->rng, uni_dist);
    return variate_generator();
  }
 private:
  boost::mt19937 rng;
};

class IntRNG {
 public:
  IntRNG() {}
  void InitRand() { this->rng.seed(static_cast<unsigned int>(std::time(0))); }
  void InitRand(unsigned int seed) { this->rng.seed(seed); }
  int Rand(int min, int max) {
    if(max < min){
        int tmp = min;
        min = max;
        max = tmp;
    }
    else if (max == min) {
      return max;
    }
    boost::uniform_int<int> uni_dist(min, max);
    boost::variate_generator<boost::mt19937&, boost::uniform_int<int> >
        variate_generator(this->rng, uni_dist);
    return variate_generator();
  }
 private:
  boost::mt19937 rng;
};

}  // namespace det_rng


namespace caffe {

/**
 * @brief Provides detection data to the Net.
 *        Source file should contain information of image paths and
 *        the corresponding label paths.
 *
 */
template <typename Dtype>
class DetectionDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit DetectionDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~DetectionDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DetectionData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }  // data&label

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleData();
  virtual void AugmentData(
    const int data_id,
    Blob<Dtype>* transformed_image_blob,
    Blob<Dtype>* transformed_label_blob);
  virtual void load_batch(Batch<Dtype>* batch);

  vector<int> data_id_vec_;  // data ids
  vector<std::string> image_path_vec_;  // image paths
  vector<std::string> label_path_vec_;  // label paths
  vector<int> bbox_num_vec_;  // number of bounding box in each image
  vector<vector<int> > bbox_cls_vec_;  // class
  vector<vector<cv::Point2f> > bbox_coord_vec_;  // central coordinates
  vector<vector<cv::Size2f> > bbox_size_vec_;  // size
  int entries_size_;  // total images
  int entries_id_;

  det_rng::FloatRNG float_rng;
  det_rng::IntRNG int_rng;
};

}  // namespace caffe

#endif  // CAFFE_DETECTION_DATA_LAYER_HPP_