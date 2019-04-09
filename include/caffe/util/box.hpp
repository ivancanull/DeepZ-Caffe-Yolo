#ifndef CAFFE_UTIL_BOX_HPP_
#define CAFFE_UTIL_BOX_HPP_

#include <algorithm>

#include "caffe/common.hpp"

namespace caffe {

template <typename Dtype>
class Box {
 public:
  Box() :  cx_(0), cy_(0), width_(0), height_(0), 
           left_(0), top_(0),  right_(0), bottom_(0) {}
  explicit Box(const Dtype cx, const Dtype cy, const Dtype width, const Dtype height)
     : cx_(cx), cy_(cy), width_(width), height_(height) {
    left_  = cx - 0.5*width;
    top_    = cy - 0.5*height;
    right_ = cx + 0.5*width;
    bottom_ = cy + 0.5*height;
  }
  virtual ~Box() {}
  bool CheckBox() const { return left_ <= right_ && top_ <= bottom_; }
  Dtype cx() const { return cx_; }
  Dtype cy() const { return cy_; }
  Dtype width() const { return width_; }
  Dtype height() const { return height_; }
  Dtype left() const { return left_; }
  Dtype top() const { return top_; }
  Dtype right() const { return right_; }
  Dtype bottom() const { return bottom_; }
  Dtype BoxArea() const { return width_*height_; }
  
 private:
  Dtype cx_;
  Dtype cy_;
  Dtype width_;
  Dtype height_;
  Dtype left_;
  Dtype top_;
  Dtype right_;
  Dtype bottom_;
};

template <typename Dtype>
Dtype calc_box_intersection(Box<Dtype> box1, Box<Dtype> box2) {
  Dtype maxx = std::min(box1.right(), box2.right());
  Dtype minx = std::max(box1.left(), box2.left());
  Dtype maxy = std::min(box1.bottom(), box2.bottom());
  Dtype miny = std::max(box1.top(), box2.top());
  Dtype intw = std::max(Dtype(0), maxx - minx);
  Dtype inty = std::max(Dtype(0), maxy - miny);
  return intw*inty;
}

template <typename Dtype>
Dtype calc_box_union(Box<Dtype> box1, Box<Dtype> box2) {
  return box1.BoxArea() + box2.BoxArea() - calc_box_intersection(box1, box2);
}

template <typename Dtype>
Dtype calc_box_iou(Box<Dtype> box1, Box<Dtype> box2) {
  if (!box1.CheckBox() || !box2.CheckBox()) 
    return Dtype(0);
  return calc_box_intersection(box1, box2) / calc_box_union(box1, box2);
}


}  // namespace caffe

#endif   // CAFFE_UTIL_BOX_HPP_