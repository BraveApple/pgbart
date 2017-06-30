#ifndef PGBART_MATH_HPP
#define PGBART_MATH_HPP

#include <vector>
#include <sstream>
#include <ctime>
#include <random>
#include <numeric>
#include <string>
#include <iostream>

#include "pgbart/include/config.hpp"
#include "pgbart/include/random.hpp"
#include "pgbart/include/compare.hpp"

/**************************************
File name : data.hpp
Date : 2016-12-7
Struct List :   Matrix - operator : (i, j)
                  (i, :)
                  (:, i)
                  + - * =
        Point
Function List : compare_point(const Point<T>& a, const Point<T>& b)
        argsort(const std::vector<T>& vec)
        unique(const std::vector<T>& vec)
        sum(const std::vector<T>& vec) -- return a number
        cumsum(const std::vector<T>& vec) -- nreturn a vector
        shuffle(const std::vector<T>& source)
        range(T start, T end, T step = 1U)
        diff(const std::vector<T>& vec)
        mean(const std::vector<T>& vec)
        check_if_included(const std::vector<T>& vec, T value)
        at(const std::map<T1, shared_ptr<T2>>& input, T1 key) -- return the element at specific position
***************************************/

// define some matrix algorithm about Matrix
namespace pgbart {

template<typename T>
struct Matrix {
  std::vector<T> elements;
  UINT n_row;
  UINT n_column;

  Matrix() : n_row(0), n_column(0) {
    this->elements = std::vector<T>();
  }

  Matrix(UINT n_row, UINT n_column) {
    this->elements = std::vector<T>(n_row * n_column, T(0));
    this->n_row = n_row;
    this->n_column = n_column;
  }

  Matrix(UINT n_row, UINT n_column, bool special) {
    if (special) {
      this->elements = std::vector<T>(n_row * n_column);
    }
    this->n_row = n_row;
    this->n_column = n_column;
  }

  Matrix(std::vector<T> vec, UINT n_row, UINT n_column) : n_row(n_row), n_column(n_column) {
    if (n_row * n_column != vec.size()) {
      std::cout << "the size of vector = " << vec.size() << " must be equal to n_row * n_column = "
        << this->n_row * this->n_column << std::endl;
      exit(1);
    }
    this->elements = vec;
  }

  // deep copy
  Matrix(const Matrix& m) {
    this->n_row = m.n_row;
    this->n_column = m.n_column;
    this->elements = m.elements;
  }

  bool hasSameSize(Matrix& m) {
    if (this->n_row == m.n_row && this->n_column = m.n_column)
      return true;
    else
      return false;
  }

  void set(UINT i, UINT j, T value) {
    if (!(i >= 0U && i < this->n_row)) {
      std::cout << "i = " << i << "must satisfy: " << i << " >= 0 && " << i << " < " << this->n_row << std::endl;
      exit(1);
    }
    if (!(j >= 0U && j < this->n_column)) {
      std::cout << "j = " << j << "must satisfy: " << j << " >= 0 && " << j << " < " << this->n_column << std::endl;
      exit(1);
    }
    this->elements[j + i * this->n_column] = value;
  }

  void set(const std::string& str, UINT j, const std::vector<T>& values) {
    if (str != ":") {
      std::cout << "the 1st parameter must be \":\"!" << std::endl;
      exit(1);
    }
    if (this->n_row != values.size()) {
      std::cout << "the size of values must be equal to n_row = " << this->n_row << std::endl;
      exit(1);
    }
    for (UINT i = 0; i < this->n_row; i++)
      set(i, j, values[i]);
  }

  void set(UINT i, const std::string& str, const std::vector<T>& values) {
    if (str != ":") {
      std::cout << "the 1st parameter must be \":\"!" << std::endl;
      exit(1);
    }
    if (this->n_column != values.size()) {
      std::cout << "the size of values must be equal to n_column = " << this->n_column << std::endl;
      exit(1);
    }
    for (UINT j = 0; j < this->n_column; j++)
      set(i, j, values[j]);
  }

  // get the element at (i, j)
  T operator() (UINT i, UINT j) {
    if (!(i >= 0U && i < this->n_row)) {
      std::cout << "i = " << i << "must satisfy: " << i << " >= 0 && " << i << " < " << this->n_row << std::endl;
      exit(1);
    }
    if (!(j >= 0U && j < this->n_column)) {
      std::cout << "j = " << j << "must satisfy: " << j << " >= 0 && " << j << " < " << this->n_column << std::endl;
      exit(1);
    }
    return this->elements[j + i * this->n_column];
  }

  // get the special element at (i, j)
  T* operator() (UINT i, UINT j, bool special) {
    if (special) {
      if (!(i >= 0U && i < this->n_row)) {
        std::cout << "i = " << i << "must satisfy: " << i << " >= 0 && " << i << " < " << this->n_row << std::endl;
        exit(1);
      }
      if (!(j >= 0U && j < this->n_column)) {
        std::cout << "j = " << j << "must satisfy: " << j << " >= 0 && " << j << " < " << this->n_column << std::endl;
        exit(1);
      }
      return &(this->elements[j + i * this->n_column]);
    }
    else
      return NULL;
  }

  // get the j-th column std::vector
  std::vector<T> operator() (const std::string& str, UINT j) {
    if (str != ":") {
      std::cout << "the 1st parameter must be \":\"!" << std::endl;
      exit(1);
    }
    if (!(j >= 0U && j < this->n_column)) {
      std::cout << "j = " << j << "must satisfy: " << j << " >= 0 && " << j << " < " << this->n_column << std::endl;
      exit(1);
    }
    std::vector<T> temp;
    for (UINT i = 0; i < this->n_row; i++)
      temp.push_back(this->elements[j + i * this->n_column]);
    return temp;
  }

  // get the j-th column std::vector
  std::vector<T> operator() (const IntVector& ids, UINT j)
  {
    std::vector<T> temp;
    if (!(j >= 0U && j < this->n_column)) {
      std::cout << "j = " << j << "must satisfy: " << j << " >= 0 && " << j << " < " << this->n_column << std::endl;
      exit(1);
    }

    for (auto i : ids) {
      if (!(i >= 0U && i < this->n_row)) {
        std::cout << "i = " << i << "must satisfy: " << i << " >= 0 && " << i << " < " << this->n_row << std::endl;
        exit(1);
      }
      temp.push_back(this->elements[j + i * this->n_column]);
    }
    return temp;
  }

  // get the i-th row std::vector
  std::vector<T> operator() (UINT i, const std::string& str) const
  {
    if (str != ":") {
      std::cout << "the 2nd parameter must be \":\"!" << std::endl;
      exit(1);
    }

    std::vector<T> temp;
    for (UINT j = 0; j < this->n_column; j++)
      temp.push_back(this->elements[j + i * this->n_column]);
    return temp;
  }

  // get the i-th row std::vector
  std::vector<T> operator() (UINT i, const IntVector& ids)
  {
    if (!(i >= 0U && i < this->n_row)) {
      std::cout << "i = " << i << "must satisfy: " << i << " >= 0 && " << i << " < " << this->n_row << std::endl;
      exit(1);
    }
    std::vector<T> temp;
    for (auto j : ids) {
      if (!(j >= 0U && j < n_column)) {
        std::cout << "j = " << j << "must satisfy: " << j << " >= 0 && " << j << " < " << this->n_column << std::endl;
        exit(1);
      }
      temp.push_back(this->elements[j + i * this->n_column]);
    }
  }

  Matrix operator() (const IntVector& ids, std::string str) {
    if (str != ":") {
      std::cout << "the 2nd parameter must be \":\"!" << std::endl;
      exit(1);
    }
    UINT n_row_temp = ids.size();
    if (n_row_temp > this->n_row) {
      std::cout << "the size of the 1st vector = " << n_row_temp << " must be less than " << this->n_row << std::endl;
      exit(1);
    }
    UINT n_column_temp = this->n_column;
    std::vector<T> elements_temp(n_row_temp * n_column_temp);
    for (UINT i = 0; i < n_row_temp; i++) {
      for (UINT j = 0; j < n_column_temp; j++)
        elements_temp[j + i * n_column_temp] = this->elements[j + ids[i] * this->n_column];
    }

    Matrix temp(elements_temp, n_row_temp, n_column_temp);
    return temp;
  }

  // overload operator "+"
  Matrix operator+ (Matrix& right) {
    if (!hasSameSize(right)) {
      std::cout << "the two matrixes must have same size!" << std::endl;
      exit(1);
    }

    Matrix temp(this->n_row, this->n_column);
    for (UINT i = 0; i < this->n_row * this->n_column; i++)
      temp.elements[i] = this->elements[i] + right.elements[i];
    return temp;
  }

  // overload operator "-"
  Matrix operator- (Matrix& right) {
    if (!hasSameSize(right)) {
      std::cout << "the two matrixes must have same size!" << std::endl;
      exit(1);
    }

    Matrix temp(this->n_row, this->n_column);
    for (UINT i = 0; i < this->n_row * this->n_column; i++)
      temp.elements[i] = this->elements[i] - right.elements[i];
    return temp;
  }

  // overload operator "*"
  Matrix operator* (T scale) {
    Matrix temp(this->n_row, this->n_column);

    for (UINT i = 0; i < this->n_row * this->n_column; i++)
      temp.elements[i] = this->elements[i] * scale;
    return temp;
  }

  // overload operator "="
//  void operator= (Matrix& m) {
  //  Matrix temp(m);
    //return temp;
//  }

  std::vector<T> toVector() {
    return this->elements;
  }

  std::vector<T> sum(UINT axis) {
    std::vector<T> sum_temp;
    if (axis == 0U) {
      for (UINT i = 0; i < this->n_row; i++) {
        T temp = T(0);
        for (auto j = 0U; j < this->n_column; j++) {
          temp += this->elements[j + i * this->n_column];
        }
        sum_temp.push_back(temp);
      }
    }
    else if (axis == 1U) {
      for (UINT j = 0; j < this->n_column; j++) {
        T temp = T(0);
        for (UINT i = 0; i < this->n_row; i++) {
          temp += this->elements[j + i * this->n_column];
        }
        sum_temp.push_back(temp);
      }
    }
    else {
      std::cout << "the axis must be zero or one!" << std::endl;
      exit(1);
    }
    return sum_temp;
  }

  std::string toString() {
    std::ostringstream os;
    for (UINT i = 0; i < this->n_row; i++) {
      os << "\n";
      for (UINT j = 0; j < this->n_column; j++)
        os << this->elements[j + i * this->n_column] << "  ";
    }
    os << "\n";
    return os.str();
  }
}; // class Matrix

} // namesapce pgbart

// ***************************************************************************
// ***************************************************************************

// define some algorithm about std::vector in STL
namespace pgbart {
namespace math {
  
template<typename T>
struct Point {
  T value;
  UINT index;
  Point() : value(T(0)), index(0) {}
  Point(UINT index, T value) : value(value), index(index) {}
};

template<typename T>
bool compare_point(const Point<T>& a, const Point<T>& b) {
  return a.value < b.value;
}

template<typename T>
IntVector argsort(const std::vector<T>& vec) {
  std::vector<Point<T>> temp(vec.size());
  for (UINT i = 0; i < vec.size(); i++) {
    Point<T> point(i, vec[i]);
    temp[i] = point;
  }
  std::sort(temp.begin(), temp.end(), compare_point<T>);

  IntVector index_vector(vec.size());
  for (UINT i = 0; i < vec.size(); i++)
    index_vector[i] = temp[i].index;
  return index_vector;
}

template<typename T>
std::vector<T> unique(const std::vector<T>& vec) {
  std::vector<T> temp;
  for (auto value : vec) {
    auto iter = find(temp.begin(), temp.end(), value);
    if (iter != temp.end()) // the value already exists in the temp
      continue;
    temp.push_back(value);
  }
  return temp;
}

// overload operator "+"
template<typename T>
std::vector<T> operator+ (const std::vector<T>& left, const std::vector<T>& right) {
  if (left.size() != right.size()) {
    std::cout << "the two vectors must have same size!" << std::endl;
    exit(1);
  }
  std::vector<T> temp;
  for (UINT i = 0; i < left.size(); i++)
    temp.push_back(left[i] + right[i]);
  return temp;
}

// overload operator "+"
template<typename T>
std::vector<T> operator+ (const std::vector<T>& left, T right) {
  std::vector<T> temp;
  for (auto left_value : left)
    temp.push_back(left_value + right);
  return temp;
}

// overload operator "+="
template<typename T>
void operator+= (std::vector<T>& left, const std::vector<T>& right) {
  left = left + right;
}

// overload operator "-"
template<typename T>
std::vector<T> operator- (const std::vector<T>& left, const std::vector<T>& right) {
  if (left.size() != right.size()) {
    std::cout << "the two vectors must have same size!" << std::endl;
    exit(1);
  }
  std::vector<T> temp;
  for (UINT i = 0; i < left.size(); i++)
    temp.push_back(left[i] - right[i]);
  return temp;
}

// overload operator "-="
template<typename T>
void operator-= (std::vector<T>& left, const std::vector<T>& right) {
  left = left - right;
}

// overload operator "-"
template<typename T>
std::vector<T> operator- (const std::vector<T>& left, T right) {
  std::vector<T> temp;
  for (UINT i = 0; i < left.size(); i++)
    temp.push_back(left[i] - right);
  return temp;
}

// overload operator "-="
template<typename T>
void operator-= (std::vector<T>& left, T right) {
  left = left - right;
}

// overload operator "*"
template<typename T>
std::vector<T> operator* (const std::vector<T>& left, T right) {
  std::vector<T> temp;
  for (UINT i = 0; i < left.size(); i++)
    temp.push_back(left[i] * right);
  return temp;
}

// overload operator "*"
template<typename T>
std::vector<T> operator* (T left, const std::vector<T>& right) {
  std::vector<T> temp;
  for (auto right_value : right)
    temp.push_back(left * right_value);
  return temp;
}

// overload operator "/"
template<typename T>
std::vector<T> operator/ (const std::vector<T>& left, T right) {
  std::vector<T> temp;
  for (auto value : left)
    temp.push_back(value / right);
  return temp;
}

template<typename T>
T sum(const std::vector<T>& vec) {
  return std::accumulate(vec.begin(), vec.end(), static_cast<T>(0.0));
}

template<typename T>
std::vector<T> cumsum(const std::vector<T>& vec) {
  T sum = T(0);
  std::vector<T> temp;
  for (auto value : vec) {
    sum += value;
    temp.push_back(sum);
  }
  return temp;
}

template<typename T>
T sum2(const std::vector<T>& vec) {
  T temp = static_cast<T>(0);
  for (auto value : vec)
    temp += value * value;
  return temp;
}

template<typename T>
std::vector<T> range(T start, T end, T step = 1U) {
  if (start > end) {
    std::cout << "the start = " << start << " must be less than the end = " << end << std::endl;
    exit(1);
  }
  if (step <= 0U) {
    std::cout << "the step must be positive!" << std::endl;
    exit(1);
  }
  std::vector<T> temp;
  for (auto i = start; i < end; i += step)
    temp.push_back(i);
  return temp;
}

template<typename T>
std::vector<T> ones(UINT length) {
  if (length < 0U) {
    std::cout << "the length must be positive!" << std::endl;
    exit(1);
  }
  std::vector<T> temp;
  for (UINT i = 0; i < length; i++)
    temp.push_back(1);
  return temp;
}

template<typename T>
std::vector<T> zeros(UINT length) {
  if (length < 0U) {
    std::cout << "the length must be positive!" << std::endl;
    exit(1);
  }
  std::vector<T> temp;
  for (UINT i = 0; i < length; i++)
    temp.push_back(T(0));
  return temp;
}

template<typename T>
std::vector<T> shuffle(const std::vector<T>& source) {
  std::vector<T> temp(source);
  std::random_shuffle(temp.begin(), temp.end());
  return temp;
}

template<typename T>
IntVector find(const std::vector<T>& vec, T element) {
  IntVector ids;
  for (UINT i = 0; i < vec.size(); i++) {
    if (vec[i] == element)
      ids.push_back(i);
  }
  return ids;
}

template<typename T>
bool delete_id(std::vector<T>& vec, UINT id) {
  if (id >= vec.size()) {
    std::cout << "beyond the upper bound!" << std::endl;
    return false;
  }
  else if (id < 0U) {
    std::cout << "below the lower bound!" << std::endl;
    return false;
  }
  else {
    vec.erase(vec.begin() + id);
    return true;
  }
}

template<typename T>
bool delete_element(std::vector<T>& vec, T element) {
  bool op = false;
  IntVector ids = find(vec, element);
  if (ids.empty())
    op = false;
  else {
    std::reverse(ids.begin(), ids.end());
    for (auto id : ids) {
      op = delete_id(vec, id);
      if (!op)
        return op;
    }
  }
  return op;
}

template<typename T>
T max(const std::vector<T>& vec) {
  auto iter = std::max_element(vec.begin(), vec.end());
  return *iter;
}

template<typename T>
T max(const T num1, const T num2) {
  return (num1 > num2 ? num1 : num2);
}

template<typename T>
T min(const std::vector<T>& vec) {
  auto iter = std::min_element(vec.begin(), vec.end());
  return *iter;
}

template<typename T>
T mean(const std::vector<T>& vec) {
  if (vec.size() <= 0U) {
    std::cout << "the length of vector must be positive!" << std::endl;
    exit(1);
  }
  T sum = std::accumulate(vec.begin(), vec.end(), static_cast<T>(0));
  return sum / vec.size();
}

template<typename T>
T variance(const std::vector<T>& vec, T mean) {
  if (vec.size() <= 1U) {
    std::cout << "the length of vector must be larger than one!" << std::endl;
    exit(1);
  }
  T sum = T(0);
  for (auto value : vec)
    sum += (value - mean) * (value - mean);
  return sum / vec.size();
}

template<typename T>
T variance(const std::vector<T>& vec) {
  if (vec.size() <= 1U) {
    std::cout << "the length of vector must be larger than one!" << std::endl;
    exit(1);
  }
  T mean_temp = mean(vec);
  return variance(vec, mean_temp);
}

template<typename T>
std::vector<T> diff(const std::vector<T>& vec) {
  if (vec.size() < 1U) {
    std::cout << "the length of vector must be larger than one!" << std::endl;
    exit(1);
  }
  if (vec.size() == 1) {
    std::vector<T> temp;
    return temp;
  }
  std::vector<T> temp;
  for (UINT i = 1; i < vec.size(); i++)
    temp.push_back(vec[i] - vec[i - 1]);
  return temp;
}

template<typename T>
std::vector<T> square(const std::vector<T>& vec) {
  std::vector<T> temp;
  for (auto value : vec)
    temp.push_back(value * value);
  return temp;
}

template<typename T>
std::vector<T> log(const std::vector<T>& vec) {
  std::vector<T> temp;
  for (auto value : vec)
    temp.push_back(std::log(value));
  return temp;
}

template<typename T>
std::vector<T> exp(const std::vector<T>& vec) {
  std::vector<T> temp;
  for (auto value : vec)
    temp.push_back(std::exp(value));
  return temp;
}

template<typename T>
T log_sum_exp(const std::vector<T>& vec) {
  T vec_max = max(vec);
  std::vector<T> vec_temp = vec - vec_max;
  return std::log(sum(exp(vec_temp))) + vec_max;
}

template<typename T>
std::vector<T> softmax(const std::vector<T>& vec) {
  T max_temp = max(vec);
  std::vector<T> temp = vec - max_temp;
  temp = exp(temp);
  return temp / sum(temp);
}

template<typename T>
void replace(std::vector<T>& output, const std::vector<T>& input, UINT start) {
  if (output.size() < input.size() + start) {
    std::cout << "output.size() < input.size() + start !" << std::endl;
    exit(1);
  }
  for (auto i = start; i < output.size(); i++)
    output[i] = input[i - start];
}

template<typename T>
std::vector<T> at(const std::vector<T>& vec, const IntVector& id_vector) {
  std::vector<T> temp;
  for (auto id : id_vector)
    temp.push_back(vec[id]);
  return temp;
}

template<typename T>
BoolVector compare_if(const std::vector<T>& vec, const std::string& compare_symbol, const T right) {
  BoolVector temp;
  for (auto left : vec)
    temp.push_back(compare::compare_if(left, compare_symbol, right));
  return temp;
}

template<typename T>
IntVector choose_ids(const std::vector<T>& vec, T split, const std::string& compare_symbol) {
  IntVector temp;
  if (compare_symbol == "<=") {
    for (UINT i = 0; i < vec.size(); i++) {
      if (vec[i] <= split)
        temp.push_back(i);
    }
  }
  else if (compare_symbol == ">") {
    for (UINT i = 0; i < vec.size(); i++) {
      if (vec[i] > split)
        temp.push_back(i);
    }
  }
  else {
    std::cout << "the 3rd parameter must be \"<=\" or \" >\"!" << std::endl;
    exit(1);
  }
  return temp;
}

template<typename T>
std::string toString(const std::vector<T>& vec) {
  std::ostringstream os;
  for (UINT i = 0; i < vec.size(); i++) {
    os << vec[i];
    if (i % 10 != 9)
      os << "  ";
    else
      os << "\n";
  }
  return os.str();
}

template<typename T>
std::string toString(const std::vector<std::vector<T>>& vec_vec) {
  std::ostringstream os;
  for (UINT i = 0; i < vec_vec.size(); i++) {
    const std::vector<T>& vec = vec_vec[i];
    std::cout << "[" << i << "]: " << std::endl;
    os << toString(vec);
  }
  return os.str();
}

template<typename T>
bool check_if_included(const std::vector<T>& vec, T value) {
  auto iter = std::find(vec.begin(), vec.end(), value);
  if (iter == vec.end())
    return false;
  else
    return true;
}

template<typename T>
std::vector<T> sort(const std::vector<T>& vec) {
  std::vector<T> temp(vec);
  std::sort(temp.begin(), temp.end());
  return temp;
}

} // namespace math

} // namesapce math

namespace pgbart {
namespace math {

template<typename T1, typename T2>
std::vector<T2> at(const std::map<T1, T2>& input, const std::vector<T1>& keys) {
  std::vector<T2> result;
  for (auto key : keys) {
    auto iter = input.find(key);
    if (iter == input.end()) {
      std::cout << "the key is not included in map!" << std::endl;
      exit(1);
    }
    result.push_back(iter->second);
  }
  return result;
}

template<typename T1, typename T2>
void initial_map(std::map<T1, T2>& input_output, const std::vector<T1>& keys, const T2& default_value) {
  input_output.clear();
  for (auto key : keys)
    input_output[key] = default_value;
}

template<typename T1, typename T2>
std::map<T1, T2> choose_pairs(const std::map<T1, T2>& input, const std::vector<T1>& keys) {
  std::map<T1, T2> result;
  for (auto key : keys) {
    auto iter = input.find(key);
    if (iter == input.end()) {
      std::cout << "the key is not included in map!" << std::endl;
      exit(1);
    }
    result[key] = iter->second;
  }
  return result;
}

template<typename T1, typename T2>
void set_keys(std::map<T1, T2>& input, const std::vector<T1>& keys) {
  for (auto key : keys)
    input[key] = T2();
}

template<typename T1, typename T2>
bool check_if_included(const std::map<T1, T2>& input, const T1& key) {
  return input.find(key) != input.end();
}

template<typename T1, typename T2>
T2 at(const std::map<T1, T2>& input, T1 key) {
  auto iter = input.find(key);
  if (iter == input.end()) {
    std::cout << "the key is not included in map!" << std::endl;
    exit(1);
  }
  return iter->second;
}

template<typename T1, typename T2>
shared_ptr<T2> at(const std::map<T1, shared_ptr<T2>>& input, T1 key) {
  auto iter = input.find(key);
  if (iter == input.end()) {
    std::cout << "the key is not included in map!" << std::endl;
    exit(1);
  }
  return iter->second;
}

template<typename T1, typename T2>
void delete_by_key(std::map<T1, T2>& input, const T1& key) {
  auto iter = input.find(key);
  if (iter == input.end()) {
    std::cout << "the key is not included in map!" << std::endl;
    exit(1);
  }
  input.erase(iter);
}

template<typename T1, typename T2>
std::vector<T1> get_keys(const std::map<T1, T2>& input) {
  std::vector<T1> temp;
  for (auto pair : input)
    temp.push_back(pair.first);
  return temp;
}

template<typename T1, typename T2>
std::string toString(const std::map<T1, T2>& input) {
  std::ostringstream os;
  UINT counter = 0;
  for (auto pair_input : input) {
    counter++;
    os << "[key = " << pair_input.first << " value = " << pair_input.second << "] ";
    if (counter % 10 == 0)
      os << std::endl;
  }
  return os.str();
}

template<typename T>
bool check_if_included(const std::set<T>& source, T value) {
  return source.end() != source.find(value);
}

template<typename T>
T absolute(T a) {
  if (a >= T(0))
    return a;
  else
    return -a;
}

template<typename T>
std::string operator* (const std::string& str, T n_time) {
  std::string str_temp = "";
  for (auto i = 0U; i < n_time; i++)
    str_temp += str;
  return str_temp;
}

} // namespace math

} // namesapce pgbart

#endif
