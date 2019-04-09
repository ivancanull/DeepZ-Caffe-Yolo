#ifdef USE_OPENCV
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/detection_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
DetectionDataLayer<Dtype>::~DetectionDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void DetectionDataLayer<Dtype>::DataLayerSetUp(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Initialize random number generator
  float_rng.InitRand();
  int_rng.InitRand();

  // Set parameters of loading image data.
  bool is_color  = this->layer_param_.detection_data_param().is_color();
  string root_folder = this->layer_param_.detection_data_param().root_folder();
  int max_bboxes = this->layer_param_.detection_data_param().max_bboxes();

  // Parse meta file and then read image paths and label paths.
  const string& meta_source = 
    this->layer_param_.detection_data_param().meta_source();
  std::ifstream infile;
  string line, image_path, label_path;
  size_t pos;
  LOG(INFO) << "Opening file: " << meta_source;
  infile.open(meta_source.c_str());
  CHECK(infile.is_open()) << "Can not open file: " << meta_source;
  while (std::getline(infile, line)) {
    pos = line.find(' ');
    image_path = line.substr(0, pos);
    label_path = line.substr(pos + 1);
    data_id_vec_.push_back(data_id_vec_.size());
    image_path_vec_.push_back(image_path);
    label_path_vec_.push_back(label_path);
  }
  infile.close();
  CHECK(!image_path_vec_.empty()) << "File is empty";
  entries_size_ = data_id_vec_.size();
  entries_id_ = 0;
  LOG(INFO) << "A total of " << entries_size_ << " images.";
  // Read all bounding boxes' label and save them in vector.
  for (int i = 0; i < label_path_vec_.size(); ++i) {
    int bbox_num = 0;
    vector<int> bbox_cls;
    vector<cv::Point2f> bbox_coord;
    vector<cv::Size2f> bbox_size;
    // Read bounding boxes' label in i-th image.
    infile.open((root_folder + label_path_vec_[i]).c_str());
    CHECK(infile.is_open()) << "Can not open file: "
      << root_folder + label_path_vec_[i];
    while (std::getline(infile, line)) {
      vector<string> bbox_label;
      boost::split(bbox_label, line, boost::is_any_of(" "));
      CHECK(5 == bbox_label.size())
        << "One bounding box label should contain 5 entries exactly";
      bbox_num += 1;
      bbox_cls.push_back(atoi(bbox_label[0].c_str()));
      bbox_coord.push_back(
        cv::Point2f(atof(bbox_label[1].c_str()), atof(bbox_label[2].c_str())));
      bbox_size.push_back(
        cv::Size2f(atof(bbox_label[3].c_str()), atof(bbox_label[4].c_str())));
    }
    infile.close();
    CHECK(bbox_num <= max_bboxes) 
      << "Number of ground-truth boxes in one image should <= " 
      << max_bboxes << "(file: " << label_path_vec_[i] << ", with "
      << bbox_num << " boxes)";
    bbox_num_vec_.push_back(bbox_num);
    bbox_cls_vec_.push_back(bbox_cls);
    bbox_coord_vec_.push_back(bbox_coord);
    bbox_size_vec_.push_back(bbox_size);
  }

  // Whether to shuffle images or not.
  if (this->layer_param_.detection_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleData();
  }

  // Set blob shape.
  DetectionDataParameter detection_data_param = 
    this->layer_param_.detection_data_param();
  const int batch_size = detection_data_param.batch_size();
  const int net_height = detection_data_param.net_height();
  const int net_width  = detection_data_param.net_width();
  // Check dimensions.
  CHECK_GT(batch_size, 0);
  CHECK_GT(net_height, 0);
  CHECK_GT(net_width, 0);
  // Build BlobShape.
  vector<int> top_shape(4);
  top_shape[0] = 1;
  top_shape[1] = is_color ? 3 : 1;
  top_shape[2] = net_height;
  top_shape[3] = net_width;
  // Set image shape.
  this->transformed_data_.Reshape(top_shape);
  top_shape[0] = batch_size;
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);
  // Set label shape.
  vector<int> label_shape(4, 1);
  label_shape[1] = max_bboxes*5;  // cls, x, y, w, h
  this->transformed_label_.Reshape(label_shape);
  label_shape[0] = batch_size;
  for (int i = 0; i < this->prefetch_.size(); ++i) {
      this->prefetch_[i]->label_.Reshape(label_shape);
  }
  top[1]->Reshape(label_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
}

template <typename Dtype>
void DetectionDataLayer<Dtype>::ShuffleData() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(data_id_vec_.begin(), data_id_vec_.end(), prefetch_rng);
}

template <typename Dtype>
void DetectionDataLayer<Dtype>::AugmentData(
    const int data_id,
    Blob<Dtype>* transformed_image_blob,
    Blob<Dtype>* transformed_label_blob) {
  // set parameter
  DetectionDataParameter detection_data_param =   
    this->layer_param_.detection_data_param();
  const bool is_color = detection_data_param.is_color();
  const string root_folder = detection_data_param.root_folder();
  const int neth = detection_data_param.net_height();
  const int netw  = detection_data_param.net_width();

  const Dtype min_scale = detection_data_param.augmentation_param().min_scale();
  const Dtype max_scale = detection_data_param.augmentation_param().max_scale();
  const bool mirror = 
    detection_data_param.augmentation_param().mirror() && int_rng.Rand(0, 1);
  const float jitter = detection_data_param.augmentation_param().jitter();
  const bool distort = detection_data_param.augmentation_param().distort();
  const float hue = detection_data_param.augmentation_param().hue();
  const float sat = detection_data_param.augmentation_param().sat();
  const float val = detection_data_param.augmentation_param().val();
  
  // Load an image.
  cv::Mat orig_img_u8 = ReadImageToCVMat(
    root_folder + image_path_vec_[data_id], is_color);
  CHECK(orig_img_u8.data) << "Could not load " << image_path_vec_[data_id];
  // convert uint8 to float32
  // Note: there may be precision loss using float32 when Dtype == double.
  //       So float64 may be a better choice? (but surely will be slower)
  /// TODO: find a better solution for this problem.
  cv::Mat orig_img;
  orig_img_u8.convertTo(orig_img, CV_32FC3, 1.0f/255.0f);

  // Data augmentation.
  // Firstly transform image and then correct the corresponding label box.

  // Transform image.
  // Randomly choose a new aspect ratio and scale.
  float ow = orig_img.cols, oh = orig_img.rows;
  float dw = jitter*ow, dh = jitter*oh;
  float rw = float_rng.Rand(-dw, dw), rh = float_rng.Rand(-dh, dh);
  float new_ar = (ow + rw) / (oh + rh);
  float scale = float_rng.Rand(min_scale, max_scale);
  float nw, nh;
  if (new_ar < 1.0f) {
    nh = scale*neth;
    nw = nh*new_ar;
  }
  else {
    nw = scale*netw;
    nh = nw/new_ar;
  }

  // Create the new image with the new aspect ratio.
  cv::Mat new_img;
  cv::resize(orig_img, new_img, cv::Size(nw, nh));

  // Crop new image to fit the input size of the network.
  // in order to make sure every single pixel in the new image can be covered
  float dx = float_rng.Rand(0.0f, netw - nw);
  float dy = float_rng.Rand(0.0f, neth - nh);
  // create and initialize a base image
  const int init_val_type = is_color ? CV_32FC3 : CV_32FC1;
  const cv::Scalar_<float> init_val = is_color ? 
      cv::Scalar_<float>(0.5f, 0.5f, 0.5f) : cv::Scalar_<float>(0.5f);
  cv::Mat sized_img(cv::Size(netw, neth), init_val_type, init_val);
  // crop
  cv::Point ul_new_img(int(std::max(-dx, 0.0f)), int(std::max(-dy, 0.0f)));
  cv::Point ul_sized_img(int(std::max(dx, 0.0f)), int(std::max(dy, 0.0f)));
  cv::Size cropped_size(
      std::min(int(nw) - ul_new_img.x, netw), 
      std::min(int(nh) - ul_new_img.y, neth));
  cv::Mat cropped_img = new_img(cv::Rect(ul_new_img, cropped_size));
  cropped_img.copyTo(sized_img(cv::Rect(ul_sized_img, cropped_size)));

  // Whether to flip the image horizontally.
  if (mirror) cv::flip(sized_img, sized_img, 1);

  // Randomly distort image
  // Note: it costs TOO MUCH time!
  //       Better implementation? Multithreading for data augmentation?
  if (distort) {
    // set distortion parameters
    float dhue = float_rng.Rand(-360.0f*hue, 360.0f*hue);
    float dsat = float_rng.Rand(1.0f/sat, sat);
    float dval = float_rng.Rand(1.0f/val, val);
    // convert from BGR space to HSV space
    cv::cvtColor(sized_img, sized_img, cv::COLOR_BGR2HSV);
    // split channels
    vector<cv::Mat> hsv_img_vec;
    cv::split(sized_img, hsv_img_vec);
    // scale hue
    for (cv::MatIterator_<float> it = hsv_img_vec[0].begin<float>(); 
        it != hsv_img_vec[0].end<float>(); ++it) {
      *it += dhue;
      if (*it > 360.0f) *it -= 360.0f;
      if (*it <   0.0f) *it += 360.0f;
    } 
    // scale saturation
    for (cv::MatIterator_<float> it = hsv_img_vec[1].begin<float>(); 
        it != hsv_img_vec[1].end<float>(); ++it) {
      *it *= dsat;
      if (*it > 1.0f) *it = 1.0f;
      if (*it < 0.0f) *it = 0.0f;
    }
    // scale value
    for (cv::MatIterator_<float> it = hsv_img_vec[2].begin<float>(); 
        it != hsv_img_vec[2].end<float>(); ++it) {
      *it *= dval;
      if (*it > 1.0f) *it = 1.0f;
      if (*it < 0.0f) *it = 0.0f;
    }
    // merge channels
    cv::merge(hsv_img_vec, sized_img);
    // convert from HSV space back to BGR space
    cv::cvtColor(sized_img, sized_img, cv::COLOR_HSV2BGR);
  }

  // Put the augmented image into top blob.
  Dtype* transformed_image = transformed_image_blob->mutable_cpu_data();
  for (int h = 0; h < neth; ++h) {
    const float* ptr = sized_img.ptr<float>(h);
    int img_index = 0;
    for (int w = 0; w < netw; ++w) {
      for (int c = 0; c < sized_img.channels(); ++c) {
        int top_index = (c*neth + h)*netw + w;
        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
        transformed_image[top_index] = pixel;
      }
    }
  }

  // Transform label.
  Dtype* transformed_label = transformed_label_blob->mutable_cpu_data();
  int bbox_num = bbox_num_vec_[data_id];
  for (int i = 0; i < bbox_num; ++i) {
    // x, y, w, h, cls
    Dtype x   = bbox_coord_vec_[data_id][i].x;
    Dtype y   = bbox_coord_vec_[data_id][i].y;
    Dtype w   = bbox_size_vec_[data_id][i].width;
    Dtype h   = bbox_size_vec_[data_id][i].height;
    Dtype cls = bbox_cls_vec_[data_id][i];

    Dtype left   = x - w/Dtype(2);
    Dtype right  = x + w/Dtype(2);
    Dtype top    = y - h/Dtype(2);
    Dtype bottom = y + h/Dtype(2);

    CHECK_LE(left, right) << "Wrong label box";
    CHECK_LE(top, bottom) << "Wrong label box";

    left   = (left  *nw + dx) / netw;
    right  = (right *nw + dx) / netw;
    top    = (top   *nh + dy) / neth;
    bottom = (bottom*nh + dy) / neth;

    if (mirror) {
      Dtype tmp = left;
      left  = Dtype(1) - right;
      right = Dtype(1) - tmp;
    }

    // constrain
    if (left   < Dtype(0)) left   = Dtype(0);
    if (left   > Dtype(1)) left   = Dtype(1);
    if (right  < Dtype(0)) right  = Dtype(0);
    if (right  > Dtype(1)) right  = Dtype(1);
    if (top    < Dtype(0)) top    = Dtype(0);
    if (top    > Dtype(1)) top    = Dtype(1);
    if (bottom < Dtype(0)) bottom = Dtype(0);
    if (bottom > Dtype(1)) bottom = Dtype(1);

    x = (left + right)/Dtype(2);
    y = (top + bottom)/Dtype(2);
    w = right - left;
    h = bottom - top;

    int top_index = i*5;
    if (w < Dtype(0.001) || h < Dtype(0.001)) {
      transformed_label[top_index + 0] = Dtype(-1);
      transformed_label[top_index + 1] = Dtype(0);
      transformed_label[top_index + 2] = Dtype(0);
      transformed_label[top_index + 3] = Dtype(0);
      transformed_label[top_index + 4] = Dtype(0);
    }
    else {
      transformed_label[top_index + 0] = cls;
      transformed_label[top_index + 1] = x;
      transformed_label[top_index + 2] = y;
      transformed_label[top_index + 3] = w;
      transformed_label[top_index + 4] = h;
    }
    
  }
  // to indicate the end fo bounding boxes
  int top_index = bbox_num*5;
  transformed_label[top_index] = Dtype(-1);
}

// This function is called on prefetch thread
template <typename Dtype>
void DetectionDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  CHECK(this->transformed_label_.count());

  // set parameters
  const int batch_size = 
    this->layer_param_.detection_data_param().batch_size();
  CHECK(batch_size == batch->data_.shape()[0]) << "batch size mismatch";

  // Currently do not support changing data dimension.

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // Get real data id.
    CHECK_GT(entries_size_, entries_id_);
    const int data_id = data_id_vec_[entries_id_];
    // Prepare for data augmentation.
    int data_offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + data_offset);
    int label_offset = batch->label_.offset(item_id);
    this->transformed_label_.set_cpu_data(prefetch_label + label_offset);
    // Load an image and do data augmentation.
    AugmentData(data_id,
      &(this->transformed_data_),
      &(this->transformed_label_));
    // Go to the next iter.
    entries_id_++;
    if (entries_id_ >= entries_size_) {
      // We have reached the end. Restart from the first.
      LOG(INFO) << "Restarting data prefetching from start.";
      entries_id_ = 0;
      if (this->layer_param_.detection_data_param().shuffle()) {
        ShuffleData();
      }
    }
  }
}

INSTANTIATE_CLASS(DetectionDataLayer);
REGISTER_LAYER_CLASS(DetectionData);

}  // namespace caffe

#endif  // USE_OPENCV