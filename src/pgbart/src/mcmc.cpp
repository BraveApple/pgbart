#include "pgbart/include/mcmc.hpp"

#include <iostream>
#include <algorithm>
#include <memory>

using namespace pgbart::math;

namespace pgbart {

Pmcmc::Pmcmc() {
  this->p_ptr = nullptr;
}

Pmcmc::Pmcmc(const Data& data, const Control& control, const Param& param, const Cache& cache, const CacheTemp& cache_temp) {
  Vec_Particle_Ptr particles_ptr;
  DoubleVector_Ptr log_weights_ptr;
  double log_pd;

  UINT n_particles_backup = control.n_particles;
  const_cast<Control&>(control).n_particles = 1;
  std::tie(particles_ptr, log_weights_ptr) = init_particles(data, control, param, cache_temp);
  const_cast<Control&>(control).n_particles = n_particles_backup;
  log_pd = -DBL_MAX;
  IntVector origin_itr = { 0 };
  for (UINT i = 0; i < particles_ptr->size(); i++) {
    (*particles_ptr)[i]->nodes_processed_itr.push_back(origin_itr);
    (*particles_ptr)[i]->grow_nodes_itr.push_back(origin_itr);
  }
  
  this->p_ptr = nullptr;
  this->update_p(particles_ptr, log_weights_ptr, log_pd, control);
}

bool Pmcmc::update_p(Vec_Particle_Ptr particles_ptr, DoubleVector_Ptr log_weights_ptr, double& log_pd, const Control& control) {

  map<UINT, SplitInfo> node_info_old;
  bool first_iter = false;
  
  if (control.verbose_level >= 2) {
    std::cout<<"log_weights = "<< toString(*log_weights_ptr) << std::endl;
  }
  UINT k = sample_multinomial_distribution(softmax(*log_weights_ptr));
  if (this->p_ptr != nullptr) {
    node_info_old = this->p_ptr->node_info;
  }
  else {
    first_iter = true;
  }
  bool same_tree = (node_info_old == (*particles_ptr)[k]->node_info);
  this->p_ptr = Particle_Ptr(new Particle(*(*particles_ptr)[k]));
  this->log_pd = log_pd;
  if (control.verbose_level >= 2) {
    std::cout << "pid_sampled = " << k << std::endl;
    std::cout << "new tree :" << std::endl;
    //this->p_ptr->print_tree();
    if (!same_tree && control.verbose_level >= 1) {
      std::cout << "non-identical trees!" << std::endl;
    }

    if (k == 0 && !first_iter)
      if (!same_tree)
        throw same_tree;
  }
  else if (same_tree && !first_iter) {
    if (control.verbose_level >= 1) {
      std::cout << "identical tree without k == 0!" << std::endl;
    }
    try {
      compare::compare_if_zero((*log_weights_ptr)[k] - (*log_weights_ptr)[0]);
    }
    catch (std::exception()) {
      throw std::exception();
    }
  }
  this->p_ptr->check_depth();
  if (control.verbose_level >= 2) {
    std::cout << "sampled particle = " << k << std::endl;
  }
  return (!same_tree);
}

bool Pmcmc::sample(const Data& data, const Control& control, const Param& param, const Cache& cache, const CacheTemp& cache_tmp) {
  // Particle Gibbs(PG) sampler

  // initialze particles and their weights
  Vec_Particle_Ptr particles_ptr;
  DoubleVector_Ptr log_weights_ptr;
  double log_pd;
  std::tie(particles_ptr, log_pd, log_weights_ptr) = init_run_smc(data, control, param, cache, cache_tmp, this->p_ptr);
  // update the tree in the mcmc_object
  bool change = update_p(particles_ptr, log_weights_ptr, log_pd, control);
  return change;
}

} // namesapce pgbart

namespace pgbart {
Pmcmc_Ptr init_tree_mcmc(const Data& data_train, const Control& control, const Param& param,
  const Cache& cache, const CacheTemp& cache_temp) {
  
  Pmcmc_Ptr pmcmc_ptr;
  pmcmc_ptr = Pmcmc_Ptr(new Pmcmc(data_train, control, param, cache, cache_temp));
  return pmcmc_ptr;
}

tuple<Particle_Ptr, bool> run_mcmc_single_tree(Particle_Ptr p_ptr, const Control& control, const Data& data_train,
  const Param& param, const Cache& cache, bool change, const CacheTemp& cache_temp, Pmcmc_Ptr pmcmc_ptr) {
  // starting mcmc
  change = pmcmc_ptr->sample(data_train, control, param, cache, cache_temp);
  // update the tree from the mcmc_object
  p_ptr = pmcmc_ptr->p_ptr;
  return make_tuple(p_ptr, change);
}

} // namespace pgbart

/*****************************
********* TreeMCMC ***********
******************************/
namespace pgbart {

IntVector_Ptr TreeMCMC::get_nodes_not_in_subtree(const int node_id) {
  IntVector_ptr reqd_nodes_ptr = make_shared<IntVector>();
  IntVector all_nodes(this->tree_ptr->leaf_node_ids);
  all_nodes.insert(all_nodes.end(), this->tree_ptr->non_leaf_node_ids.begin(),
    this->tree_ptr->non_leaf_node_ids.end());
  IntVector_Ptr subtree_ptr = this->get_nodes_subtree(node_id);
  for (auto it = all_nodes.begin(); it != all_nodes.end(); it++) {
    auto iter = std::find(subtree_ptr->begin(), subtree_ptr->end(), *it);
    if (iter == subtree_ptr.end()) { // Not found node in subtree
      reqd_nodes_ptr->push_back(*it);
    }
  }
  return reqd_nodes_ptr;
}

IntVector_Ptr TreeMCMC::get_nodes_subtree(const int node_id) {
  // NOTE: current node_id is included in nodes_subtree as well
  IntVector_Ptr node_list_ptr = make_shared<IntVector>();
  IntVector expand {node_id};
  while (expand.size() > 0) {
    const int node = expand.back();
    expand.pop_back();
    node_list_ptr->push_back(node);
    if (!this->tree_ptr->isLeafNode(node)) {
      const int left  = this->tree_ptr->getLeftNodeID(node);
      const int right = this->tree_ptr->getRightNodeID(node);
      expand.push_back(left);
      expand.push_back(right);
    }
  }
  return node_list_ptr;
}

double TreeMCMC::compute_log_acc_g(const int node_id, const Param& param, const int len_both_children_terminal, 
  const double loglik, const IntVector& train_ids_left, const IntVector& train_ids_right, 
  const Cache& cache, const Control& control, const Data& data, const IntVector& grow_nodes) {
  // Effect of do_not_split dose not matter for node_id since it has chidren
  double logprior_children = 0.0;
  const int left = this->tree_ptr->getLeftNodeID(node_id);
  const int right = this->tree_ptr->getRightNodeID(node_id);

  if (!no_valid_split_exists(data, cache, train_ids_left)) {
    logprior_children += log(compute_not_split_prob(this->tree_ptr, left, param));
  }
  if (!no_valid_split_exists(data, cache, train_ids_right)) {
    logprior_children += log(compute_not_split_prob(this->tree_ptr, right, param));
  }

  const double log_acc_prior = log(compute_split_prob(this->tree_ptr, node_id, param))
    - log(compute_not_split_prob(this->tree_ptr, node_id, param))
    - log(len_both_children_terminal) + log(grow_nodes.size()) + logprior_children;
  const double log_acc_loglik = loglik - this->loglik[node_id];
  const double log_acc = log_acc_prior + log_acc_loglik;

  if (loglik == -DBL_MAX) // Just need to ensure that an invalid split is not grown
    log_acc = -DBL_MAX;

  return log_acc;
}

double TreeMCMC::compute_log_inv_acc_p(const int node_id, const Param& param, const int len_both_children_terminal,
  const double loglik, const IntVector& grow_nodes, const Cache& cache, const Control& control, 
  const Data& train_data) {
  // acc for Grow except for corrections to both_children_terminal and grow_nodes list
  double logprior_children = 0;
  const int left = this->tree_ptr->getLeftNodeID(node_id);
  const int right = this->tree_ptr->getRightNodeID(node_id);

  if (!no_valid_split_exists(data, cache, this->train_ids[left])) {
    logprior_children += log(compute_not_split_prob(this->tree_ptr, left, param));
  }

  if (!no_valid_split_exists(data, cache, this->train_ids[right])) {
    logprior_children += log(compute_not_split_prob(this->tree_ptr, right, param));
  }

  if (compare_if_zero(logprior_children - this->log_prior[left] - this->log_prior[right])) {
    std::cout << "oh oh ... looks like a bug in compute_log_inv_acc_p" << std::endl;
    exit(1);
  }

  const double log_inv_acc_prior = log(compute_split_prob(this->tree_ptr, node_id, param))
    - log(compute_not_split_prob(this->tree_ptr, node_id, param))
    - log(len_both_children_terminal) + log(grow_nodes.size()) + logprior_children;
  const double log_inv_acc_loglik = loglik - this->loglik[node_id];
  const double log_inv_acc = log_inv_acc_loglik + log_inv_acc_prior;

  if (log_inv_acc > -DBL_MAX) {
    std::cout << "Error" << std::endl;
  }
  return log_inv_acc;
}

bool TreeMCMC::grow(const Data& train_data, const Control& control, const Param& param, const Cache& cache,
    const IntVector& grow_nodes) {
  bool change = false;
  if (grow_nodes.size() == 0)
    return change;
  const int node_id = ramdom_choice(grow_nodes);
  bool do_not_split_node_id; SplitInfo split_info; double logprior_nodeid;
  tie(do_not_split_node_id, &split_info, logprior_nodeid) = 
    this->sample_split_prior(node_id, train_data, param, control, cache);
  if (!do_not_split_node_id) {
    std::cout << "Error: do_not_split_node_id" << std::endl;
    exit(1);
  }
  const IntVector train_ids = this->train_ids[node_id];
  IntVector train_ids_left; IntVector train_ids_right; CacheTemp cache_temp;
  tie(&train_ids_left, &train_ids_right, &cache_temp) = 
    this->compute_left_right_statistics(train_data, param, control, cache, 
    feat_id_chosen, split_chosen, train_ids);
  const double loglik = cache_temp.loglik_left + cache_temp.loglik_right;
  const int len_both_children_terminal_new = this->both_children_terminal.size();
  const int sibling_node_id = this->tree_ptr->getSiblingNodeID(node_id);
  if (!this->tree_ptr->isLeafNode(sibling_node_id)) {
    len_both_children_terminal_new += 1;
  }
  const double log_acc = this->compute_log_acc_g(node_id, param, len_both_children_terminal_new, 
    loglik, train_ids_left, train_ids_right, cache, control, train_data, grow_nodes);
  const double log_r = log(simulate_continuous_uniform_distribution(0, 1));
  if (log_r <= log_acc) {
    this->update_left_right_statistics(node_id, logprior_nodeid, split_info, cache_temp, control, 
      train_ids_left, train_ids_right, train_data, cache, param);
    // MCMC specific data structure updates
    this->both_children_terminal.push_back(node_id);
    const int parent_id = this->tree_ptr->getParentNodeID(node_id);
    if (node_id != 0 && this->tree_ptr->isNonLeafNode(node_id)) {
      self.inner_pc_pairs.push_back(make_tuple(parent_id, node_id));
    }
    if (this->tree_ptr->isLeafNode(sibling_node_id)) {
      math::delete_element<int>(this->both_children_terminal, parent_id);
    }
    change = true;
  } else {
    change = false;
  }
  return change;
}

bool TreeMCMC::prune(const Data& train_data, const Control& control, const Param& param, const Cache& cache,
    const IntVector& grow_nodes) {
  bool change = false;
  if (this->both_children_terminal.size() == 0) {
    return change;
  }
  const int node_id = ramdom_choice(grow_nodes);
  const int feat_id = this->node_info[node_id].feat_id_chosen;
  const int left = this->tree_ptr->getLeftNodeID(node_id);
  const int right = this->tree_ptr->getRightNodeID(node_id);
  const double loglik = this->loglik[left] + self.loglik[right];
  int len_both_children_terminal_new = this->both_children_terminal.size();
  IntVector grow_nodes_temp(grow_nodes);
  grow_nodes_temp.push_back(node_id);
  if (math::check_if_included<int>(grow_nodes_temp, left)) {
    math::delete_element<int>(grow_nodes_temp, left);
  }
  if (math::check_if_included<int>(grow_nodes_temp, right)) {
    math::delete_element<int>(grow_nodes_temp, right);
  }
  const double log_acc = -1.0 * this->compute_log_inv_acc_p(node_id, param, len_both_children_terminal,
    loglik, grow_nodes_temp, cache, control, train_data);
  const double log_r = log(simulate_continuous_uniform_distribution(0, 1));
  if (log_r <= log_acc) {
    this->remove_leaf_node_statistics(left);
    this->remove_leaf_node_statistics(right);
    this->tree_ptr->addLeafNode(node_id);
    this->tree_ptr->removeNonLeafNode(node_id);
    this->log_prior[node_id] = log(compute_not_split_prob(this->tree_ptr, node_id, param));
    // OK to set logprior as above since we know that a valid split exists
    // MCMC specific data structure updates
    math::delete_element<int>(this->both_children_terminal, node_id);
    parent = this->tree_ptr->getParentNodeID(node_id);
    if (node_id != 0 && this->tree_ptr->isNonLeafNode(node_id)) {
      // math::delete_element<tuple<int, int>>(this->inner_pc_pairs, make_tuple(parent, node_id));
      math::delete_element<NodePair>(this->inner_pc_pairs, NotePair(parent, node_id));
    }
    if (node_id != 0) {
      const int sibling_node_id = this->tree_ptr->getSiblingNodeID(node_id);
      if (this->tree_ptr->isLeafNode(sibling_node_id)) {
        this->both_children_terminal.push_back(parent)
      }
    }
    change = true;
  } else {
    change = false;
  }
  return change;
}

bool TreeMCMC::change(const Data& train_data, const Control& control, const Param& param, const Cache& cache,
    const IntVector& grow_nodes) {
  bool change = false;
  if (this->tree_ptr->non_leaf_nodes.size()) {
    return change;
  }
  const int node_id = ramdom_choice(grow_nodes);
  bool do_not_split_node_id; SplitInfo split_info; double logprior_nodeid;
  tie(do_not_split_node_id, &split_info, logprior_nodeid) = 
    this->sample_split_prior(node_id, train_data, param, control, cache);
  // Note: this just samples a split criterion, not guaranteed to "change"
  if (do_not_split_node_id) {
    std::cout << "do not split node id" << std::endl;
  }
  IntVector_Ptr nodes_subtree_ptr = this->get_nodes_subtree(node_id);
  IntVector_Ptr nodes_not_in_subtree_ptr = this->get_nodes_not_in_subtree(node_id);
  // self.create_new_statistics(nodes_subtree, nodes_not_in_subtree, node_id, settings)
  this->create_new_statistics(nodes_subtree_ptr, nodes_not_in_subtree_ptr, node_id);
  this->node_info_new[node_id] = this->node_info[node_id];
  // self.evaluate_new_subtree(data, node_id, param, nodes_subtree, cache, settings)
  this->evaluate_new_subtree(train_data, node_id, param, nodes_subtree_ptr, cache);
  // log_acc will be modified below
  double log_acc_temp = 0; double loglik_diff = 0; logprior_diff = 0;
  // log_acc_temp, loglik_diff, logprior_diff = self.compute_log_acc_cs(nodes_subtree, node_id)
  const double log_acc = log_acc_temp = log_acc_temp + this->log_prior[node_id];
    - this->logprior_new[node_id];
  const double log_r = log(simulate_continuous_uniform_distribution(0, 1));
  if (log_r <= log_acc) {
    this->node_info[node_id] = this->node_info_new[node_id];
    // self.update_subtree(node_id, nodes_subtree, settings)
    change = true;
  } else {
    change = false;
  }
  return change;
}

bool TreeMCMC::swap(const Data& train_data, const Control& control, const Param& param, const Cache& cache,
  const IntVector& grow_nodes) {
  bool change = false;
  if (this->inner_pc_pairs.size() == 0) {
    return change;
  }
  const int id = simulate_discrete_uniform_distribution(0, this->inner_pc_pairs.size() - 1);
  int node_id = 0; int child_id = 0;
  tie(node_id, child_id) = this->inner_pc_pairs[id]; // (parent, child) pair
  IntVector_Ptr nodes_subtree_ptr = this->get_nodes_subtree(node_id);
  IntVector_Ptr nodes_not_in_subtree_ptr = this->get_nodes_not_in_subtree(node_id);
  // self.create_new_statistics(nodes_subtree, nodes_not_in_subtree, node_id, settings)
  this->create_new_statistics(nodes_subtree_ptr, nodes_not_in_subtree_ptr, node_id);
  this->node_info_new[node_id] = this->node_info[node_id];
  this->node_info_new[node_id] = this->node_info[child_id];
  // self.evaluate_new_subtree(data, node_id, param, nodes_subtree, cache, settings)
  this->evaluate_new_subtree(train_data, node_id, param, nodes_subtree_ptr, cache);
  double log_acc = 0; double loglik_diff = 0; logprior_diff = 0;
  // log_acc, loglik_diff, logprior_diff = self.compute_log_acc_cs(nodes_subtree, node_id)
  const double log_r = log(simulate_continuous_uniform_distribution(0, 1));
  if (log_r <= log_acc) {
    this->node_info[node_id] = this->node_info_new[node_id];
    this->node_info[child_id] = this->node_info_new[child_id];
    // self.update_subtree(node_id, nodes_subtree, settings)
    change = true;
  } else {
    change =false;
  }
  return change;
}

tuple<bool, MoveType> TreeMCMC::sample(const Data& train_data, const Control& control, const Param& param, 
  const Cache& cache) {
  MoveType move_type = simulate_discrete_uniform_distribution(0, 3);
  double log_acc = -DBL_MAX;
  double log_r = 0.0;
  IntVector grow_nodes;
  for (UINT i = 0; i < this->tree_ptr->leaf_node_ids.size(); i++) {
    const int leaf_node_id = this->tree_ptr->leaf_node_ids[i];
    if (!stop_split(this->train_ids[leaf_node_id], control, train_data, cache)) {
      grow_nodes.push_back(leaf_node_id);
    }
  }
  bool change = false;
  switch(move_type) {
    case GROW: change = this->grow(train_data, control, param, cache, grow_nodes);
    case PRUNE: change = this->prune(train_data, control, param, cache, grow_nodes);
    case CHANGE: change = this->change(train_data, control, param, cache, grow_nodes);
    case SWAP: change = this->swap(train_data, control, param, cache, grow_nodes);
    default: { std::cout << "Error move type!" << std::endl; exit(1); } 
  }
  if (change) {
    this->tree_ptr->updateTreeDepth();
    this->loglik_current = 0.0;
    for (auto node_id : this->tree_ptr->leaf_node_ids) {
      this->loglik_current += this->loglik[node_id];
    }
  }
  return make_tuple(change, move_type);
}

} // namesapce pgbart

    
