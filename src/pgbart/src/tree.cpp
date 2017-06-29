#include <algorithm>
#include <iostream>

#include "pgbart/include/tree.hpp"
#include "pgbart/include/random.hpp"
#include "pgbart/include/math.hpp"

namespace pgbart {
Tree::Tree() {
  this->tree_depth = -1;
  this->root_node_id = 0;
  this->total_node.clear();
  this->leaf_node_ids.clear();
  this->non_leaf_node_ids.clear();
  this->pred_val_n.clear();
}

Tree::Tree(const shared_ptr<Tree> other) {
  this->tree_depth = other->tree_depth;
  this->root_node_id = other->root_node_id;
  this->total_node = other->total_node;
  this->leaf_node_ids = other->leaf_node_ids;
  this->non_leaf_node_ids = other->non_leaf_node_ids;
  this->pred_val_n = other->pred_val_n;
}

shared_ptr<Node> Tree::getRoot() {
  if (math::check_if_included(this->total_node, 0U)) {
    std::cout << "node_id = 0 is not a node of the tree!" << std::endl;
    exit(1);
  }
  return this->total_node[0];
}

bool Tree::isLeafNode(UINT node_id) const {
  return math::check_if_included(this->leaf_node_ids, node_id);
}

bool Tree::isNonLeafNode(UINT node_id) const {
  return math::check_if_included(this->non_leaf_node_ids, node_id);
}

void Tree::addLeafNode(UINT node_id) {
  if (this->isLeafNode(node_id)) {
    std::cout << "node_id = " << node_id << " is a ready leaf node.";
    exit(1);
  }
  this->leaf_node_ids.puhs_back(node_id);
}

void Tree::removeLeafNode(UINT node_id) {
  if (!this->isLeafNode(node_id)) {
    std::cout << "node_id = " << node_id << " is not a leaf node.";
    exit(1);
  }
  math::delete_element<int>(this->leaf_node_ids, node_id);
}

void Tree::addNonLeafNode(UINT node_id) {
  if (this->isNonLeafNode(node_id)) {
    std::cout << "node_id = " << node_id << " is a ready non-leaf node.";
    exit(1);
  }
  this->non_leaf_node_ids.puhs_back(node_id);
}

void Tree::removeNonLeafNode(UINT node_id) {
  if (!this->isNonLeafNode(node_id)) {
    std::cout << "node_id = " << node_id << " is not a non-leaf node.";
    exit(1);
  }
  math::delete_element<int>(this->non_leaf_node_ids, node_id);
}

UINT Tree::getLeftNodeID(UINT node_id) const {
  if (math::check_if_included(this->total_node, node_id)) {
    std::cout << "node_id = "<< node_id << " is not a node of the tree!" << std::endl;
    exit(1);
  }
  UINT left_node_id = 2 * node_id + 1;
  if (math::check_if_included(this->total_node, left_node_id)) {
    std::cout << "node_id = " << node_id << " is a leaf node!" << std::endl;
    exit(1);
  }
  return left_node_id;
}

UINT Tree::getRightNodeID(UINT node_id) const {
  if (math::check_if_included(this->total_node, node_id)) {
    std::cout << "node_id = "<< node_id << " is not a node of the tree!" << std::endl;
    exit(1);
  }
  UINT right_node_id = 2 * node_id + 2;
  if (math::check_if_included(this->total_node, right_node_id)) {
    std::cout << "node_id = " << node_id << " is a leaf node!" << std::endl;
    exit(1);
  }
  return right_node_id;
}

UINT Tree::getParentNodeID(UINT node_id) {
  if (math::check_if_included(this->total_node, node_id)) {
    std::cout << "node_id = "<< node_id << " is not a node of the tree!" << std::endl;
    exit(1);
  }
  if (node_id == 0) {
    std::cout << "node_id = " << node_id << " is a root node, so it does not have parent node!" << std::endl;
    exit(1);
  }
  UINT parent_id = static_cast<UINT>(std::ceil(node_id / 2.0) - 1);
  return parent_id;
}

UINT Tree::getSiblingNodeID(UINT node_id) const {
  if (math::check_if_included(this->total_node, node_id)) {
    std::cout << "node_id = "<< node_id << " is not a node of the tree!" << std::endl;
    exit(1);
  }
  if (node_id == 0) {
    std::cout << "node_id = " << node_id << " is a root node, so it does not have parent node!" << std::endl;
    exit(1);
  }
  const int parent = this->getParentNodeID(node_id);
  const int left = this->getLeftNodeID(parent);
  const int right = this->getRightNodeID(parent);
  int sibling_id = 0;
  if (left == node_id) {
    sibling_id = right;
  }
  if (right == node_id) {
    sibling_id = left;
  }
  return sibling_id;
}
 
UINT Tree::getNodeDepth(UINT node_id) const {
  if (math::check_if_included(this->total_node, node_id)) {
    std::cout << "node_id = "<< node_id << " is not a node of the tree!" << std::endl;
    exit(1);
  }
  return (UINT)std::floor(std::log2(node_id + 1));
}

UINT Tree::getTreeDepth() {
  return this->tree_depth;
}

void Tree::updateTreeDepth() {
  UINT newest_node_id = math::max(this->leaf_node_ids);
  this->tree_depth = this->getNodeDepth(newest_node_id);
}

bool Tree::goLeft(UINT node_id, const DoubleVector& single_data) {
  if (math::check_if_included(this->total_node, node_id)) {
    std::cout << "node_id = "<< node_id << " is not a node of the tree!" << std::endl;
    exit(1);
  }
  const shared_ptr<Node> node_ptr = this->total_node[node_id];
  UINT feat_id_chosen = node_ptr->get_feat_id_chosen();
  return node_ptr->goLeft(single_data[feat_id_chosen]);
}

UINT Tree::traverse(const DoubleVector& single_data) {
  UINT node_id = 0;
  while (true) {
    if (this->isLeafNode(node_id))
      break;
    UINT left_child_id = this->getLeftNodeID(node_id);
    UINT right_child_id = this->getRightNodeID(node_id);
    node_id = this->goLeft(node_id, single_data) ? left_child_id : right_child_id;
  }
  return node_id;
}

IntVector Tree::traverse(const Matrix<double>& x) {
  IntVector leaf_node_ids(x.n_row, 0);
  for (UINT point_id = 0; point_id < x.n_row; point_id++) {
    UINT leaf_node_id = this->traverse(x(point_id, ":"));
    leaf_node_ids[point_id] = leaf_node_id;
  }
  return leaf_node_ids;
}

DoubleVector Tree::pred_real_val(const Data& data_test) {
  // aggregate prediction vals outside and use lambda_bart to compute posterior
  DoubleVector pred_val(data_test.n_point, 0.0);
  const Matrix<double>& x_test = data_test.x;
  //UINT n_point = x_test.n_row;
  //UINT n_feature = x_test.n_column;
  for (UINT i = 0; i < data_test.n_point; i++) {
    const DoubleVector& x_ = x_test(i, ":");
    UINT node_id = traverse(x_);
    pred_val[i] = pred_val_n[node_id];
  }
  return pred_val;
}
}
