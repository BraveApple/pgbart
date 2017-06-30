#ifndef PGBART_DATA_HPP
#define PGBART_DATA_HPP

#include <iostream>

#include "pgbart/include/math.hpp"

/**************************************
File name : data.hpp
Date : 2016-12-7
Struct List : Data
***************************************/

namespace pgbart {
struct Data {
  Matrix<double> x;
  DoubleVector y_residual;
  DoubleVector y_original;
  UINT n_point;
  UINT n_feature;
  Data(const Matrix<double>& x, const DoubleVector& y_original) : x(x), y_original(y_original) {
    if (x.n_row != y_original.size()) {
      std::cout << "faul to create class Data!" << std::endl;
      exit(1);
    }
    this->n_point = x.n_row;
    this->n_feature = x.n_column;
  }

  Data(){}

  void operator= (Data& another) {
    this->n_point = another.n_point;
    this->n_feature = another.n_feature;
    this->x = another.x;
    this->y_original = another.y_original;
  }
};

} // namespace pgbart

#endif
