#ifndef PGBART_SERIALIZE_HPP
#define PGBART_SERIALIZE_HPP

#include <fstream>
#include <iostream>

#include "pgbart/include/particle.hpp"
#include "pgbart/include/config.hpp"

/**************************************
File name : serialize.hpp
Date : 2016-12-7
Struct List : Model -- used to save or load the result
Function List : save_model(const string& savepath, const Model& model)
        load_model(const string& savepath)
***************************************/

namespace pgbart {

struct BartTree{
  IntVector leaf_node_ids;
  IntVector non_leaf_node_ids;
  map<UINT, double> pred_val;
  map<UINT, SplitInfo> node_info;
};

struct Model {
double train_mean;
  UINT m_bart;
UINT n_iteration;
UINT current_itr;
vector<vector<BartTree>> trees;

Model(double _mean, UINT _m_bart, UINT _n_iteration) : train_mean(_mean), m_bart(_m_bart), trees(_n_iteration), n_iteration(_n_iteration), current_itr(0) {}

void add_itr(Bart& bart, const Control& control){
  this->trees[current_itr].resize(m_bart);
  if (control.mcmc_type == "pg") {
    for (UINT i = 0; i < m_bart; i++) {
      this->trees[current_itr][i].leaf_node_ids = bart.p_particles[i]->tree_ptr->leaf_node_ids;
      this->trees[current_itr][i].non_leaf_node_ids = bart.p_particles[i]->tree_ptr->non_leaf_node_ids;
      this->trees[current_itr][i].pred_val = bart.p_particles[i]->tree_ptr->pred_val_n;
      this->trees[current_itr][i].node_info = bart.p_particles[i]->node_info;
    }
    current_itr++;
  }
  else if (control.mcmc_type == "cgm") {
    for (UINT i = 0; i < m_bart; i++) {
      this->trees[current_itr][i].leaf_node_ids = bart.p_treemcmcs[i]->tree_ptr->leaf_node_ids;
      this->trees[current_itr][i].non_leaf_node_ids = bart.p_treemcmcs[i]->tree_ptr->non_leaf_node_ids;
      this->trees[current_itr][i].pred_val = bart.p_treemcmcs[i]->tree_ptr->pred_val_n;
      this->trees[current_itr][i].node_info = bart.p_treemcmcs[i]->node_info;
    }
    current_itr++;
  }
}

};

void save_model(const string& savepath, const Model& model){
  std::ofstream output(savepath);
  if (!output) {
    std::cerr << "save error!" << std::endl;
    abort();
  }
output << model.train_mean << std::endl;
  UINT m_bart = model.m_bart;
  output << m_bart << std::endl;
UINT n_iteration = model.n_iteration;
output << n_iteration << std::endl;
for (UINT itr = 0; itr < n_iteration; itr++){
  for (UINT m = 0; m < m_bart; m++){
    BartTree& tree = const_cast<Model&>(model).trees[itr][m];
    UINT leaf_size = tree.leaf_node_ids.size();
    UINT non_leaf_size = tree.non_leaf_node_ids.size();
    output << leaf_size << std::endl;
    for (auto t : tree.leaf_node_ids){
      output << t << " " << tree.pred_val[t] << std::endl;
    }
    output << non_leaf_size << std::endl;
    for (auto t : tree.non_leaf_node_ids){
      output << t << " " << tree.node_info[t].feat_id_chosen
        << " " << tree.node_info[t].split_chosen
        << " " << tree.node_info[t].idx_split_global << std::endl;
    }
  }
}
  output.close();
}

Model load_model(const string& savepath){
  std::ifstream input(savepath);
  if (!input){
    std::cerr << "load error" << std::endl;
    abort();
  }

  UINT m_bart, n_iteration;
  double train_mean;
input >> train_mean;
  input >> m_bart;
input >> n_iteration;
  Model model(train_mean, m_bart, n_iteration);
for (UINT itr = 0; itr < n_iteration; itr++){
  model.trees[itr].resize(m_bart);
  for (UINT m = 0; m < m_bart; m++){
    BartTree& tree = model.trees[itr][m];
    UINT leaf_size;
    UINT non_leaf_size;
    input >> leaf_size;
    tree.leaf_node_ids.resize(leaf_size);
    for (UINT i = 0; i < leaf_size; i++){
      input >> tree.leaf_node_ids[i];
      input >> tree.pred_val[tree.leaf_node_ids[i]];
    }
    input >> non_leaf_size;
    tree.non_leaf_node_ids.resize(non_leaf_size);
    for (UINT i = 0; i < non_leaf_size; i++){
      input >> tree.non_leaf_node_ids[i];
      UINT feat_id_chosen;
      double split_chosen;
      UINT idx_split_chosen;
      input >> feat_id_chosen;
      input >> split_chosen;
      input >> idx_split_chosen;
      SplitInfo info(feat_id_chosen, split_chosen, idx_split_chosen);
      tree.node_info[tree.non_leaf_node_ids[i]] = info;
    }
  }
}
  input.close();
  return model;
}

} // namespace pgbart

#endif
