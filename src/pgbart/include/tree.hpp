#ifndef PGBART_TREE_HPP
#define PGBART_TREE_HPP

#include <memory>

#include "pgbart/include/node.hpp"
#include "pgbart/include/config.hpp"
#include "pgbart/include/data.hpp"

namespace pgbart {
class Tree;
typedef std::shared_ptr<Tree> Tree_Ptr;

class Tree {  
public:
  /*
  the key of map may be not continuous, if the deep of right sub-tree is greater than left sub-tree.
  for example, total_node = {(0, Node0), (1, Node1), (2, Node2), (5, Node3), (6, Node4)}, where Node0 is
  root node; Node1 and Node2 are the left and right child node of Node0 respectively; Node3 and Node
  are the left and right child node of Node2 respectively. Thus the keys of parent and children will have:
  Equation[1]: left_child_node_id = 2 * parent_node_id + 1;
  Equation[2]: right_child_node_id = 2 * parent_node_id + 2;
  Equation[3]: parent_node_id = ceil(child_node_id / 2) - 1;
  */
  //map<UINT, Node*> total_node; // key = node_id, value = *Node
  map<UINT, shared_ptr<Node>> total_node; // key = node_id, value = *Node
  IntVector leaf_node_ids; // store the id of leaf node
  IntVector non_leaf_node_ids;// store the id of non-leaf node
  map<UINT, double> pred_val_n; // store mu of leaf node
  int tree_depth; // the deep of the tree
  UINT root_node_id;

public:

  Tree();

  Tree(const shared_ptr<Tree> other);
  
  shared_ptr<Node> getRoot();

  bool isLeafNode(UINT node_id) const;

  bool isNonLeafNode(UINT node_id) const;

  void addLeafNode(UINT node_id);

  void removeLeafNode(UINT node_id);

  void addNonLeafNode(UINT node_id);

  void removeNonLeafNode(UINT node_id);
  
  UINT getLeftNodeID(UINT node_id) const;

  UINT getRightNodeID(UINT node_id) const;

  UINT getParentNodeID(UINT node_id) const;

  UINT getSiblingNodeID(UINT node_id) const;

  UINT getTreeDepth();

  UINT getNodeDepth(UINT node_id) const;

  void updateTreeDepth();

  bool goLeft(UINT node_id, const DoubleVector& single_data);

  UINT traverse(const DoubleVector& single_data);

  IntVector traverse(const Matrix<double>& x);

  DoubleVector pred_real_val(const Data& data_test);
};
}

#endif