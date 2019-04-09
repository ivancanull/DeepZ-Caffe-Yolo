#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/layers/yolo_detection_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
inline Dtype sigmoid_act(Dtype x) {
  return Dtype(1) / (Dtype(1) + exp(-x));
}

template <typename Dtype>
inline Dtype sigmoid_grad(Dtype fx) {
  return fx*(Dtype(1) - fx);
}

template <typename Dtype>
void YoloDetectionLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // set loss weight
  CHECK_EQ(top.size(), 3);
  CHECK_LE(this->layer_param_.loss_weight_size(), top.size());
  while (this->layer_param_.loss_weight_size() < top.size()) {
    this->layer_param_.add_loss_weight(Dtype(1));
  }
  // clear first
  anchor_vec_.clear();
  classes_vec_.clear();
  loss_type_vec_.clear();
  threshold_vec_.clear();
  shape_vec_.clear();
  stage_vec_.clear();
  object_scale_vec_.clear();
  noobject_scale_vec_.clear();
  coord_scale_vec.clear();
  class_scale_vec_.clear();
  // set parameters
  YoloDetectionLossParameter det_loss_param =
      this->layer_param_.yolo_detection_loss_param();
  bottom_num_ = det_loss_param.loss_param_size();
  CHECK_EQ(bottom_num_, bottom.size() - 1);
  for (int i = 0; i < bottom_num_; ++i) {
    YoloLossParameter loss_param = det_loss_param.loss_param(i);
    // retrieve anchors and classes
    YoloDetectionParameter det_param = loss_param.detection_param();
    vector<pair<Dtype, Dtype> > anchor_lst;
    for (int j = 0; j < det_param.anchors_size(); ++j) {
      AnchorParameter anchor = det_param.anchors(j);
      anchor_lst.push_back(make_pair(anchor.width(), anchor.height()));
    }
    anchor_vec_.push_back(anchor_lst);
    classes_vec_.push_back(det_param.classes());
    CHECK_EQ(classes_vec_[i], classes_vec_[0]) 
        << "Number of class should be the same.";
    // retrieve other parameters
    vector<YoloLossType> loss_type_lst;
    loss_type_lst.push_back(loss_param.confidence_loss_type());
    loss_type_lst.push_back(loss_param.position_loss_type());
    loss_type_lst.push_back(loss_param.class_loss_type());
    loss_type_vec_.push_back(loss_type_lst);
    threshold_vec_.push_back(loss_param.ignore_threshold());
    object_scale_vec_.push_back(loss_param.object_scale());
    noobject_scale_vec_.push_back(loss_param.noobject_scale());
    coord_scale_vec.push_back(loss_param.coord_scale());
    class_scale_vec_.push_back(loss_param.class_scale());
    // retrieve layer shape
    vector<int> shape_lst;
    shape_lst.push_back(bottom[i]->shape(0));
    shape_lst.push_back(bottom[i]->shape(1));
    shape_lst.push_back(bottom[i]->shape(2));
    shape_lst.push_back(bottom[i]->shape(3));
    shape_vec_.push_back(shape_lst);
    // initialize stage vector
    stage_vec_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>()));
  }
}

template <typename Dtype>
void YoloDetectionLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // loss layer output scalars; 0 axes
  vector<int> loss_shape(0);
  for (int l = 0; l < top.size(); ++l) {
    top[l]->Reshape(loss_shape);
  }
  // check
  CHECK_EQ(bottom_num_, bottom.size() - 1);  // can not change bottom_num_
  CHECK_EQ(bottom.back()->count(2), 1);
  for (int l = 0; l < bottom_num_; ++l) {
    // check batch and channel
    CHECK_EQ(bottom[l]->shape(0), bottom[0]->shape(0));
    CHECK_EQ(bottom[l]->shape(1), (5 + classes_vec_[l])*anchor_vec_[l].size());
    // update layer shape
    shape_vec_[l][0] = bottom[l]->shape(0);
    shape_vec_[l][1] = bottom[l]->shape(1);
    shape_vec_[l][2] = bottom[l]->shape(2);
    shape_vec_[l][3] = bottom[l]->shape(3);
    // reshape stage
    stage_vec_[l]->ReshapeLike(*bottom[l]);
  }
}

template <typename Dtype>
void YoloDetectionLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Get ground truth bounding boxes.
  const Blob<Dtype>* label_blob = bottom.back();
  const Dtype* label = label_blob->cpu_data();
  // <class, coordinates+shape>
  vector<vector<pair<int, Box<Dtype> > > > label_box_vec(label_blob->shape(0));
  for (int b = 0; b < label_blob->shape(0); ++b) {
    for (int n = 0; n < label_blob->shape(1); n += 5) {
      int index = label_blob->shape(1)*b + n;
      if (label[index + 0] < Dtype(0)) 
        break;
      label_box_vec[b].push_back(make_pair(
          label[index + 0], Box<Dtype>(label[index + 1], label[index + 2], 
          label[index + 3], label[index + 4])));
    }
  }

  // Calculate loss and diff.
  for (int l = 0; l < bottom_num_; ++l) {  // for each bottom data layer
    // set up and initialize
    const Dtype* data_array = bottom[l]->cpu_data();
    Dtype* loss_array = stage_vec_[l]->mutable_cpu_data();
    Dtype* diff_array = stage_vec_[l]->mutable_cpu_diff();
    caffe_set(stage_vec_[l]->count(), Dtype(0), loss_array);
    caffe_set(stage_vec_[l]->count(), Dtype(0), diff_array);
    //
    for (int b = 0; b < shape_vec_[l][0]; ++b) {
      // step 1
      for (int n = 0; n < anchor_vec_[l].size(); ++n) {
        for (int y = 0; y < shape_vec_[l][2]; ++y) {
          for (int x = 0; x < shape_vec_[l][3]; ++x) {
            Box<Dtype> pred_box = GetPredictionBox(l, b, n, y, x, data_array);
            // find the best IoU with label boxes of the current pred box
            Dtype best_iou = Dtype(0);
            for (int i = 0; i < label_box_vec[b].size(); ++i) {
              Box<Dtype> label_box = label_box_vec[b][i].second;
              Dtype temp_iou = calc_box_iou<Dtype>(label_box, pred_box);
              best_iou = std::max(best_iou, temp_iou);
            }
            // whether to ignore calculating loss
            if (best_iou < threshold_vec_[l]) {
              int index = IndexEntry(l, b, (5 + classes_vec_[l])*n, y, x);
              CalcConfidenceLossDiff(l, index, noobject_scale_vec_[l],
                data_array, Dtype(0), loss_array, diff_array);
            }
          }
        }
      }
      // step 2
      for (int i = 0; i < label_box_vec[b].size(); ++i) {
        int label_class = label_box_vec[b][i].first;
        Box<Dtype> label_box = label_box_vec[b][i].second;
        // locate
        int x = int(label_box.cx()*shape_vec_[l][3]);
        int y = int(label_box.cy()*shape_vec_[l][2]);
        // match the best anchor through all anchors
        Box<Dtype> shift_box(Dtype(0), Dtype(0),
          label_box.width(), label_box.height());
        Dtype best_iou = Dtype(0);
        pair<int, int> best_anchor;
        for (int t = 0; t < anchor_vec_.size(); ++t){
          for (int n = 0; n < anchor_vec_[t].size(); ++n) {
            Box<Dtype> anchor_box(Dtype(0), Dtype(0),
              anchor_vec_[t][n].first, anchor_vec_[t][n].second);
            Dtype temp_iou = calc_box_iou<Dtype>(shift_box, anchor_box);
            if (temp_iou > best_iou) {
              best_iou = temp_iou;
              best_anchor = make_pair(t, n);
            }
          }
        }
        // whether to calculate loss/diff (matched anchor in current l)
        if (best_anchor.first == l) {
          // re-calculate confidence loss and diff
          int confidence_index = 
            IndexEntry(l, b, (5+classes_vec_[l])*best_anchor.second+0, y, x);
          CalcConfidenceLossDiff(l, confidence_index, object_scale_vec_[l],
            data_array, Dtype(1), loss_array, diff_array);
          // calculate position loss and diff
          int position_index = 
            IndexEntry(l, b, (5+classes_vec_[l])*best_anchor.second+1, y, x);
          CalcPositionLossDiff(l, position_index, y, x, best_anchor.second, 
            coord_scale_vec[l]*(Dtype(2)-label_box.width()*label_box.height()), data_array, label_box, loss_array, diff_array);
          // calculate class loss and diff
          if (classes_vec_[l]) {
            int class_index = 
              IndexEntry(l, b, (5+classes_vec_[l])*best_anchor.second+5, y, x);
            CalcClassLossDiff(l, class_index, class_scale_vec_[l], data_array,
              label_class, loss_array, diff_array);
          } 
        }
      }
    }
  }

  // calculate total loss and set top blob
  CalcTotalLoss(
      top[0]->mutable_cpu_data(), 
      top[1]->mutable_cpu_data(),
      top[2]->mutable_cpu_data());
}

template <typename Dtype>
void YoloDetectionLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, 
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down.back()) {
    LOG(FATAL) << this->type()
      << " Layer cannot backpropagate to label inputs.";
  }
  for (int l = 0; l < bottom_num_; ++l) {
    if (propagate_down[l]) {
      const Dtype* src_array = stage_vec_[l]->cpu_diff();
      Dtype* des_array = bottom[l]->mutable_cpu_diff();
      caffe_copy(bottom[l]->count(), src_array, des_array);
    }

    /*
    LOG(INFO) << "data: " << bottom[l]->cpu_data()[0];
    LOG(INFO) << "loss: " << stage_vec_[l]->cpu_data()[0];
    LOG(INFO) << "diff: " << bottom[l]->cpu_diff()[0];
    
    for (int n = 0; n < anchor_vec_[0].size(); ++n) {
      printf("------------------- box %d -------------------\n", n);
      printf("confidence\n");
      for (int y = 0; y < shape_vec_[l][2]; ++y) {
        for (int x = 0; x < shape_vec_[l][3]; ++x) {
          int idx = IndexEntry(l, 0, 5*n, y, x);
          printf("%6.3f ", sigmoid_act(bottom[l]->cpu_data()[idx]));
        }
        printf("\n");
      }
      printf("loss\n");
      for (int y = 0; y < shape_vec_[l][2]; ++y) {
        for (int x = 0; x < shape_vec_[l][3]; ++x) {
          int idx = IndexEntry(l, 0, 5*n, y, x);
          printf("%6.3f ", stage_vec_[l]->cpu_data()[idx]);
        }
        printf("\n");
      }
      printf("diff\n");
      for (int y = 0; y < shape_vec_[l][2]; ++y) {
        for (int x = 0; x < shape_vec_[l][3]; ++x) {
          int idx = IndexEntry(l, 0, 5*n, y, x);
          printf("%6.3f ", bottom[l]->cpu_diff()[idx]);
        }
        printf("\n");
      }
      printf("------------------- box %d -------------------\n\n", n);
    }
    */ 
  }
}

template <typename Dtype>
int YoloDetectionLossLayer<Dtype>::IndexEntry(int l, int b, int c, int y, int x)
    const {
  CHECK_LE(l, shape_vec_.size());
  return ((b*shape_vec_[l][1] + c)*shape_vec_[l][2] + y)*shape_vec_[l][3] + x;
}

template <typename Dtype>
Box<Dtype> YoloDetectionLossLayer<Dtype>::GetPredictionBox(int l, int b, int n, 
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

template <typename Dtype>
void YoloDetectionLossLayer<Dtype>::CalcConfidenceLossDiff(int l, int index, 
    Dtype alpha, const Dtype* data_array, Dtype label, Dtype* loss_array, 
    Dtype* diff_array) {
  // data in data_array have not been activated
  Dtype data = sigmoid_act<Dtype>(data_array[index]);
  if (L2Loss == loss_type_vec_[l][0]) {
    loss_array[index] = alpha*caffe_l2_loss(label, data);
    diff_array[index] = 
      alpha*caffe_l2_diff(label, data)*sigmoid_grad<Dtype>(data);
  }
  else {
    /* No Implementation */
  }
}

template <typename Dtype>
void YoloDetectionLossLayer<Dtype>::CalcPositionLossDiff(int l, int index, 
    int y, int x, int n, Dtype alpha, const Dtype* data_array, 
    Box<Dtype> label_box, Dtype* loss_array, Dtype* diff_array) {
  // encode ground truth box first
  Dtype tx = shape_vec_[l][3]*label_box.cx() - x;
  Dtype ty = shape_vec_[l][2]*label_box.cy() - y;
  Dtype tw = log(label_box.width()/anchor_vec_[l][n].first);
  Dtype th = log(label_box.height()/anchor_vec_[l][n].second);

  // activate data
  int step = shape_vec_[l][2]*shape_vec_[l][3];
  Dtype data0 = sigmoid_act<Dtype>(data_array[index + 0*step]);
  Dtype data1 = sigmoid_act<Dtype>(data_array[index + 1*step]);
  Dtype data2 = data_array[index + 2*step];
  Dtype data3 = data_array[index + 3*step];

  // calculate position loss and diff
  if (L2Loss == loss_type_vec_[l][1]) {
    // L2 loss
    loss_array[index + 0*step] = alpha*caffe_l2_loss(tx, data0);
    loss_array[index + 1*step] = alpha*caffe_l2_loss(ty, data1);
    loss_array[index + 2*step] = alpha*caffe_l2_loss(tw, data2);
    loss_array[index + 3*step] = alpha*caffe_l2_loss(th, data3);

    // L2 diff
    diff_array[index + 0*step] = 
      alpha*caffe_l2_diff(tx, data0)*sigmoid_grad<Dtype>(data0);
    diff_array[index + 1*step] = 
      alpha*caffe_l2_diff(ty, data1)*sigmoid_grad<Dtype>(data1);
    diff_array[index + 2*step] = alpha*caffe_l2_diff(tw, data2);
    diff_array[index + 3*step] = alpha*caffe_l2_diff(th, data3);
  }
  else {
    /* No Implementation */
  }
}

template <typename Dtype>
void YoloDetectionLossLayer<Dtype>::CalcClassLossDiff(int l, int index,
    Dtype alpha, const Dtype* data_array, int label_cls, Dtype* loss_array, 
    Dtype* diff_array) {
  int step = shape_vec_[l][2]*shape_vec_[l][3];
  if (L2Loss == loss_type_vec_[l][2]) {
    if (loss_array[index]) {  // re-calculate class loss
      Dtype data = sigmoid_act(data_array[index + label_cls*step]);
      loss_array[index + label_cls*step] = 
        alpha*caffe_l2_loss(Dtype(1), data);
      diff_array[index + label_cls*step] = 
        alpha*caffe_l2_diff(Dtype(1), data)*sigmoid_grad(data);
    }
    else {  // first time to reach this index
      for (int c = 0; c < classes_vec_[l]; ++c) {
        Dtype data = sigmoid_act(data_array[index + c*step]);
        loss_array[index + c*step] =
          alpha*caffe_l2_loss(Dtype(c == label_cls), data);
        diff_array[index + c*step] =
          alpha*caffe_l2_diff(Dtype(c == label_cls), data)*sigmoid_grad(data);
      }
    }
  }
  else {
    /* No Implementation */
  }
}

template <typename Dtype>
void YoloDetectionLossLayer<Dtype>::CalcTotalLoss(Dtype* confidence_loss, 
    Dtype* position_loss, Dtype* class_loss) {
  *confidence_loss = Dtype(0);
  *position_loss   = Dtype(0);
  *class_loss      = Dtype(0);
  for (int l = 0; l < bottom_num_; ++l) {
    const Dtype* loss_array = stage_vec_[l]->cpu_data();
    int step = shape_vec_[l][2]*shape_vec_[l][3];
    for (int b = 0; b < shape_vec_[l][0]; ++b) {
      for (int n = 0; n < anchor_vec_[l].size(); ++n) {
        // confidence loss
        int confidence_index = IndexEntry(l, b, (5+classes_vec_[l])*n+0, 0, 0);
        for (int i = 0; i < step; ++i)
          *confidence_loss += loss_array[confidence_index + i];
        // position loss
        int position_index = IndexEntry(l, b, (5+classes_vec_[l])*n+1, 0, 0);
        for (int i = 0; i < 4*step; ++i)
          *position_loss += loss_array[position_index + i];
        // class loss
        int class_index = IndexEntry(l, b, (5+classes_vec_[l])*n+5, 0, 0);
        for (int i = 0; i < classes_vec_[l]*step; ++i)
          *class_loss += loss_array[class_index + i];
      }
    }
  }
  *confidence_loss /= shape_vec_[0][0];
  *position_loss   /= shape_vec_[0][0];
  *class_loss      /= shape_vec_[0][0];
}


INSTANTIATE_CLASS(YoloDetectionLossLayer);
REGISTER_LAYER_CLASS(YoloDetectionLoss);

}  // namespace caffe