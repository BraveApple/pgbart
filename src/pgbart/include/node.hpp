#ifndef PGBART_NODE_HPP
#define PGBART_NODE_HPP

#include "pgbart/include/config.hpp"

namespace pgbart {
class Node {
private:

  UINT feat_id_chosen; // the feature which the node choose to split
  double split_value_chosen; // the spliting value of the chosen feature

public:

  Node() : feat_id_chosen(0U), split_value_chosen(0){}

  Node(UINT feat_id_chosen, double split_chosen) : 
    feat_id_chosen(feat_id_chosen), split_value_chosen(split_chosen){}

  void initialize(UINT feat_id_chosen, double split_value_chosen) {
    this->feat_id_chosen = feat_id_chosen;
    this->split_value_chosen = split_value_chosen;
  }

  UINT get_feat_id_chosen() const {
    return feat_id_chosen;
  }

  double get_split_value_chosen() const {
    return split_value_chosen;
  }
  
  bool goLeft(double feat_value) const {
    return feat_value <= split_value_chosen;
  }

};
}

#endif