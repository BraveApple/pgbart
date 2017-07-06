#include <iostream>
#include <algorithm>
#include <memory>

#include "pgbart/include/mcmc.hpp"

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
  log_pd = -BART_DBL_MAX;
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
Pmcmc_Ptr init_particle_mcmc(const Data& data_train, const Control& control, const Param& param,
  const Cache& cache, const CacheTemp& cache_temp) {

  // Pmcmc_Ptr pmcmc_ptr;
  // pmcmc_ptr = Pmcmc_Ptr(new Pmcmc(data_train, control, param, cache, cache_temp));
  Pmcmc_Ptr pmcmc_ptr = make_shared<Pmcmc>(data_train, control, param, cache, cache_temp);
  return pmcmc_ptr;
}

tuple<Particle_Ptr, bool> run_particle_mcmc_single_tree(const Control& control,
  const Data& data_train, const Param& param, const Cache& cache, bool change,
  const CacheTemp& cache_temp, Pmcmc_Ptr pmcmc_ptr) {
  // starting mcmc
  change = pmcmc_ptr->sample(data_train, control, param, cache, cache_temp);
  // update the tree from the mcmc_object
  Particle_Ptr p_ptr = pmcmc_ptr->p_ptr;
  return make_tuple(p_ptr, change);
}

} // namespace pgbart

/*****************************
********* TreeMCMC ***********
******************************/
namespace pgbart {

TreeMCMC::TreeMCMC(const IntVector& train_ids, const Param& param, const Control& control,
  const CacheTemp& cache_temp): State(train_ids, param, cache_temp) {

  this->inner_pc_pairs.clear(); // list of nodes where both parent/child are non-terminal
  this->both_children_terminal.clear();

  this->node_info_new.clear();
  this->loglik_new.clear();
  this->logprior_new.clear();
  this->train_ids_new.clear();
  this->sum_y_new.clear();
  this->sum_y2_new.clear();
  this->n_points_new.clear();
  this->mu_mean_post_new.clear();
  this->mu_prec_post_new.clear();
}

IntVector_Ptr TreeMCMC::get_nodes_not_in_subtree(const int node_id) {
  IntVector_Ptr reqd_nodes_ptr = make_shared<IntVector>();
  IntVector all_nodes(this->tree_ptr->leaf_node_ids);
  all_nodes.insert(all_nodes.end(), this->tree_ptr->non_leaf_node_ids.begin(),
    this->tree_ptr->non_leaf_node_ids.end());
  IntVector_Ptr subtree_ptr = this->get_nodes_subtree(node_id);
  for (auto it = all_nodes.begin(); it != all_nodes.end(); it++) {
    auto iter = std::find(subtree_ptr->begin(), subtree_ptr->end(), *it);
    if (iter == subtree_ptr->end()) { // node not find in subtree
      reqd_nodes_ptr->push_back(*it);
    }
  }
  return reqd_nodes_ptr;
}

IntVector_Ptr TreeMCMC::get_nodes_subtree(const int node_id) {
  // NOTE: current node_id is included in nodes_subtree as well
  IntVector_Ptr node_list_ptr = make_shared<IntVector>();
  IntVector expand { static_cast<UINT>(node_id) };
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
    logprior_children += std::log(compute_not_split_prob(this->tree_ptr, left, param));
  }
  if (!no_valid_split_exists(data, cache, train_ids_right)) {
    logprior_children += std::log(compute_not_split_prob(this->tree_ptr, right, param));
  }

  const double log_acc_prior = std::log(compute_split_prob(this->tree_ptr, node_id, param))
    - std::log(compute_not_split_prob(this->tree_ptr, node_id, param))
    - std::log(len_both_children_terminal) + std::log(grow_nodes.size()) + logprior_children;
  const double log_acc_loglik = loglik - this->loglik[node_id];
  double log_acc = log_acc_prior + log_acc_loglik;

  if (loglik == -BART_DBL_MAX) // Just need to ensure that an invalid split is not grown
    log_acc = -BART_DBL_MAX;

  return log_acc;
}

double TreeMCMC::compute_log_inv_acc_p(const int node_id, const Param& param, const int len_both_children_terminal,
  const double loglik, const IntVector& grow_nodes, const Cache& cache, const Control& control,
  const Data& train_data) {
  // acc for Grow except for corrections to both_children_terminal and grow_nodes list
  double logprior_children = 0;
  const int left = this->tree_ptr->getLeftNodeID(node_id);
  const int right = this->tree_ptr->getRightNodeID(node_id);

  if (!no_valid_split_exists(train_data, cache, this->train_ids[left])) {
    logprior_children += std::log(compute_not_split_prob(this->tree_ptr, left, param));
  }

  if (!no_valid_split_exists(train_data, cache, this->train_ids[right])) {
    logprior_children += std::log(compute_not_split_prob(this->tree_ptr, right, param));
  }

  if (!compare::compare_if_zero(logprior_children - this->logprior[left] - this->logprior[right])) {
    std::cout << "oh oh ... looks like a bug in compute_log_inv_acc_p" << std::endl;
    exit(1);
  }

  const double log_inv_acc_prior = std::log(compute_split_prob(this->tree_ptr, node_id, param))
    - std::log(compute_not_split_prob(this->tree_ptr, node_id, param))
    - std::log(len_both_children_terminal) + std::log(grow_nodes.size()) + logprior_children;
  const double log_inv_acc_loglik = loglik - this->loglik[node_id];
  const double log_inv_acc = log_inv_acc_loglik + log_inv_acc_prior;

  if (log_inv_acc <= -BART_DBL_MAX) {
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

  std::cout << "grow_nodes = " << grow_nodes[0] << "\n\n";
  std::cout << "random_choice_id = " << node_id << "\n\n";

  bool do_not_split_node_id; SplitInfo_Ptr split_info_ptr; double logprior_nodeid;
  tie(do_not_split_node_id, split_info_ptr, logprior_nodeid) =
    this->sample_split_prior(node_id, train_data, param, control, cache);
  if (do_not_split_node_id) {
    std::cout << "Error: do_not_split_node_id" << std::endl;
    exit(1);
  }
  const IntVector train_ids = this->train_ids[node_id];
  const int feat_id_chosen = split_info_ptr->feat_id_chosen;
  const double split_chosen = split_info_ptr->split_chosen;
  IntVector_Ptr train_ids_left_ptr; IntVector_Ptr train_ids_right_ptr; CacheTemp_Ptr cache_temp_ptr;

  tie(train_ids_left_ptr, train_ids_right_ptr, cache_temp_ptr) =
    compute_left_right_statistics(train_data, param, control, cache,
    feat_id_chosen, split_chosen, train_ids);

  const double loglik = cache_temp_ptr->loglik_left + cache_temp_ptr->loglik_right;
  int len_both_children_terminal_new = this->both_children_terminal.size();
  const int sibling_node_id = this->tree_ptr->getSiblingNodeID(node_id);
  if (!this->tree_ptr->isLeafNode(sibling_node_id)) {
    len_both_children_terminal_new += 1;
  }
  const double log_acc = this->compute_log_acc_g(node_id, param, len_both_children_terminal_new,
    loglik, *train_ids_left_ptr, *train_ids_right_ptr, cache, control, train_data, grow_nodes);
  const double log_r = std::log(simulate_continuous_uniform_distribution(0, 1));

  if (log_r <= log_acc) {
    this->update_left_right_statistics(node_id, logprior_nodeid, *split_info_ptr, *cache_temp_ptr, 
      control, *train_ids_left_ptr, *train_ids_right_ptr, train_data, cache, param);
    // MCMC specific data structure updates
    this->both_children_terminal.push_back(node_id);
    const int parent_id = this->tree_ptr->getParentNodeID(node_id);
    if (node_id != 0 && this->tree_ptr->isNonLeafNode(parent_id)) {
      this->inner_pc_pairs.push_back(NodePair(parent_id, node_id));
    }
    if (this->tree_ptr->isLeafNode(sibling_node_id)) {
      math::delete_element<UINT>(this->both_children_terminal, parent_id);
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
  const int node_id = ramdom_choice(this->both_children_terminal);
  // const int feat_id = this->node_info[node_id].feat_id_chosen;
  const int left = this->tree_ptr->getLeftNodeID(node_id);
  const int right = this->tree_ptr->getRightNodeID(node_id);
  const double loglik = this->loglik[left] + this->loglik[right];
  int len_both_children_terminal_new = this->both_children_terminal.size();
  IntVector grow_nodes_temp(grow_nodes);
  grow_nodes_temp.push_back(node_id);
  if (math::check_if_included<UINT>(grow_nodes_temp, left)) {
    math::delete_element<UINT>(grow_nodes_temp, left);
  }
  if (math::check_if_included<UINT>(grow_nodes_temp, right)) {
    math::delete_element<UINT>(grow_nodes_temp, right);
  }
  const double log_acc = -1.0 * this->compute_log_inv_acc_p(node_id, param, len_both_children_terminal_new,
    loglik, grow_nodes_temp, cache, control, train_data);
  const double log_r = std::log(simulate_continuous_uniform_distribution(0, 1));
  if (log_r <= log_acc) {
    this->remove_leaf_node_statistics(left);
    this->remove_leaf_node_statistics(right);
    this->tree_ptr->addLeafNode(node_id);
    this->tree_ptr->removeNonLeafNode(node_id);
    this->logprior[node_id] = std::log(compute_not_split_prob(this->tree_ptr, node_id, param));
    // OK to set logprior as above since we know that a valid split exists
    // MCMC specific data structure updates
    math::delete_element<UINT>(this->both_children_terminal, node_id);
    const int parent = this->tree_ptr->getParentNodeID(node_id);
    if (node_id != 0 && this->tree_ptr->isNonLeafNode(parent)) {
      math::delete_element<NodePair>(this->inner_pc_pairs, NodePair(parent, node_id));
    }
    if (node_id != 0) {
      const int sibling_node_id = this->tree_ptr->getSiblingNodeID(node_id);
      if (this->tree_ptr->isLeafNode(sibling_node_id)) {
        this->both_children_terminal.push_back(parent);
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
  if (this->tree_ptr->non_leaf_node_ids.size() == 0) {
    return change;
  }
  const int node_id = ramdom_choice(this->tree_ptr->non_leaf_node_ids);
  bool do_not_split_node_id; SplitInfo_Ptr split_info_ptr; double logprior_nodeid;
  tie(do_not_split_node_id, split_info_ptr, logprior_nodeid) =
    this->sample_split_prior(node_id, train_data, param, control, cache);
  // Note: this just samples a split criterion, not guaranteed to "change"
  if (do_not_split_node_id) {
    std::cout << "do not split node id" << std::endl;
    exit(1);
  }
  IntVector_Ptr nodes_subtree_ptr = this->get_nodes_subtree(node_id);
  IntVector_Ptr nodes_not_in_subtree_ptr = this->get_nodes_not_in_subtree(node_id);
  this->create_new_statistics(*nodes_subtree_ptr, *nodes_not_in_subtree_ptr);
  this->node_info_new[node_id] = this->node_info[node_id];
  this->evaluate_new_subtree(train_data, node_id, param, *nodes_subtree_ptr, cache, control);
  // log_acc will be modified below
  double log_acc_temp = 0; double loglik_diff = 0; double logprior_diff = 0;
  tie(log_acc_temp, loglik_diff, logprior_diff) = this->compute_log_acc_cs(*nodes_subtree_ptr);
  const double log_acc = log_acc_temp + this->logprior[node_id] - this->logprior_new[node_id];
  const double log_r = std::log(simulate_continuous_uniform_distribution(0, 1));
  if (log_r <= log_acc) {
    this->node_info[node_id] = this->node_info_new[node_id];
    this->update_subtree(*nodes_subtree_ptr);
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
  const int node_id = this->inner_pc_pairs[id].parent;
  const int child_id = this->inner_pc_pairs[id].child;
  IntVector_Ptr nodes_subtree_ptr = this->get_nodes_subtree(node_id);
  IntVector_Ptr nodes_not_in_subtree_ptr = this->get_nodes_not_in_subtree(node_id);
  this->create_new_statistics(*nodes_subtree_ptr, *nodes_not_in_subtree_ptr);
  this->node_info_new[node_id] = this->node_info[child_id];
  this->node_info_new[child_id] = this->node_info[node_id];
  this->evaluate_new_subtree(train_data, node_id, param, *nodes_subtree_ptr, cache, control);
  double log_acc = 0; double loglik_diff = 0; double logprior_diff = 0;
  tie(log_acc, loglik_diff, logprior_diff) = this->compute_log_acc_cs(*nodes_subtree_ptr);
  const double log_r = std::log(simulate_continuous_uniform_distribution(0, 1));
  if (log_r <= log_acc) {
    this->node_info[node_id] = this->node_info_new[node_id];
    this->node_info[child_id] = this->node_info_new[child_id];
    this->update_subtree(*nodes_subtree_ptr);
    change = true;
  } else {
    change =false;
  }
  return change;
}

bool TreeMCMC::sample(const Data& train_data, const Control& control, const Param& param,
  const Cache& cache) {

  const int move_type = simulate_discrete_uniform_distribution(0, 3);

  std::cout << "\n\nmove_type = " << move_type << "\n\n";

  // double log_acc = -BART_DBL_MAX;
  // double log_r = 0.0;
  IntVector grow_nodes;
  for (UINT i = 0; i < this->tree_ptr->leaf_node_ids.size(); i++) {
    const int leaf_node_id = this->tree_ptr->leaf_node_ids[i];
    if (!stop_split(this->train_ids[leaf_node_id], control, train_data, cache)) {
      grow_nodes.push_back(leaf_node_id);
    }
  }
  bool change = false;

  std::cout << "\n\ngrow_nodes.size = " << grow_nodes.size() << "\n\n";

  switch(move_type) {
    case 0: { change = this->grow(train_data, control, param, cache, grow_nodes); break; }
    case 1: { change = this->prune(train_data, control, param, cache, grow_nodes); break; }
    case 2: { change = this->change(train_data, control, param, cache, grow_nodes); break; }
    case 3: { change = this->swap(train_data, control, param, cache, grow_nodes); break; }
    default: { std::cout << "Error move type!" << std::endl; exit(1); }
  }


  if (change) {
    this->tree_ptr->updateTreeDepth();
    this->loglik_current = 0.0;
    for (auto node_id : this->tree_ptr->leaf_node_ids) {
      this->loglik_current += this->loglik[node_id];
    }
  }


  return change;
}

bool TreeMCMC::check_if_same(const double log_acc, const double loglik_diff, const double logprior_diff) {
	
  double loglik_diff_2, logprior_diff_2, log_acc_2;
	UINT leaf_length = this->tree_ptr->leaf_node_ids.size();
	double sum1 = 0, sum2 = 0;
	for (UINT i = 0; i < leaf_length; i++) {
		UINT node_id = this->tree_ptr->leaf_node_ids[i];
		sum1 += this->loglik_new[node_id];
		sum2 += this->loglik[node_id];
	}
	loglik_diff_2 = sum1 - sum2;
	map<UINT, double>::iterator it;
	it = this->logprior_new.begin();
	sum1 = 0;
	sum2 = 0;
	while (it != this->logprior_new.end()) {
		sum1 += it->second;
		it++;
	}
	it = this->logprior.begin();
	while (it != this->logprior.end()) {
		sum2 += it->second;
		it++;
	}
	logprior_diff_2 = sum1 - sum2;
	log_acc_2 = loglik_diff_2 + logprior_diff_2;

  bool op = false;
	if (fabs(log_acc_2) < 1e-10) {
    op = true;
  } else {
		if (log_acc != -BART_DBL_MAX && log_acc_2 == -BART_DBL_MAX) {
			std::cout << "check_if_terms_match" << std::endl;
			std::cout << "loglik_diff = " << loglik_diff << ", " << "loglik_diff_2 = " << loglik_diff_2 << std::endl;
			std::cout << "logprior_diff = " << logprior_diff << ", " << "logprior_diff_2 = " << logprior_diff_2 << std::endl;
			op = false;
		}
	}
  return op;
}

tuple<double, double, double> TreeMCMC::compute_log_acc_cs(const IntVector& nodes_subtree){
	//for change or swap operations
	double loglik_old = 0.0; double loglik_new = 0.0; double loglik_diff = 0.0;
  double logprior_old = 0.0; double logprior_new = 0.0; 
  double logprior_diff = 0.0; double log_acc = 0.0;
	
  UINT subtree_length = nodes_subtree.size();
	// double sum_loglik_old = 0, sum_loglik_new = 0, sum_prior_old = 0, sum_prior_new = 0;
  double sum_loglik_old = 0, sum_loglik_new = 0;
	for (UINT i = 0; i < subtree_length; i++){
		UINT node_id = nodes_subtree[i];
		if (check_if_included(this->tree_ptr->leaf_node_ids, node_id)){
			sum_loglik_old += this->loglik[node_id];
			sum_loglik_new += this->loglik_new[node_id];
		}
		logprior_old += this->logprior[node_id];
		logprior_new += this->logprior_new[node_id];
	}
	loglik_diff = loglik_new - loglik_old;
	logprior_diff = logprior_new - logprior_old;
	log_acc = loglik_diff + logprior_diff;
	return make_tuple(log_acc, loglik_diff, logprior_diff);
}

void TreeMCMC::create_new_statistics(const IntVector& nodes_subtree, const IntVector& nodes_not_in_subtree) {
	this->node_info_new = this->node_info;
	UINT not_length = nodes_not_in_subtree.size();
	for (UINT i = 0; i < not_length; i++){
		UINT node_id = nodes_not_in_subtree[i];
		this->loglik_new[node_id] = this->loglik[node_id];
		this->logprior_new[node_id] = this->logprior[node_id];
		this->train_ids_new[node_id] = this->train_ids[node_id];
		this->sum_y_new[node_id] = this->sum_y[node_id];
		this->sum_y2_new[node_id] = this->sum_y2[node_id];
		this->n_points_new[node_id] = this->n_points[node_id];
		this->mu_mean_post_new[node_id] = this->mu_mean_post[node_id];
		this->mu_prec_post_new[node_id] = this->mu_prec_post[node_id];
	}
	UINT in_length = nodes_subtree.size();
	for (UINT i = 0; i < in_length; i++){
		UINT node_id = nodes_subtree[i];
		this->loglik_new[node_id] = -BART_DBL_MAX;
		this->logprior_new[node_id] = -BART_DBL_MAX;
		this->train_ids_new[node_id].clear();
		this->sum_y_new[node_id] = 0;
		this->sum_y2_new[node_id] = 0;
		this->n_points_new[node_id] = 0;
		this->mu_mean_post_new[node_id] = this->mu_mean_post[node_id];
		this->mu_prec_post_new[node_id] = this->mu_prec_post[node_id];
	}
}

void TreeMCMC::evaluate_new_subtree(const Data& train_data, const UINT node_id_start, const Param& param,
  const IntVector& nodes_subtree, const Cache& cache, const Control& control){

  for (UINT i : this->train_ids[node_id_start]){
		const DoubleVector x_ = train_data.x(i, ":");
		const double y_ = train_data.y_original[i];
		UINT node_id = node_id_start;
		while (true){
			this->sum_y_new[node_id] += y_;
			this->sum_y2_new[node_id] += y_ * y_;
			this->n_points_new[node_id] += 1;
			this->train_ids_new[node_id].push_back(i);
			if (check_if_included(this->tree_ptr->leaf_node_ids, node_id))
				break;
			UINT left = 2 * node_id + 1;
			UINT right = left + 1;
			UINT feat_id = this->node_info_new[node_id].feat_id_chosen;
			double split = this->node_info_new[node_id].split_chosen;
			// UINT idx_split_global = this->node_info_new[node_id].idx_split_global;
			if (x_[feat_id] <= split)
				node_id = left;
			else
				node_id = right;
		}
	}

	// CacheTemp_Ptr cache_temp_ptr(new CacheTemp());
	CacheTemp_Ptr cache_temp_ptr = make_shared<CacheTemp>();
  for (UINT node_id : nodes_subtree){
		this->loglik_new[node_id] = -BART_DBL_MAX;
		if (this->n_points_new[node_id] > 0){

			cache_temp_ptr->n_points = this->n_points_new[node_id];
			cache_temp_ptr->sum_y = this->sum_y_new[node_id];
			cache_temp_ptr->sum_y2 = this->sum_y2_new[node_id];
			compute_normal_normalizer(param, cache, *cache_temp_ptr, "parent");
			this->loglik_new[node_id] = cache_temp_ptr->loglik;
			this->mu_mean_post[node_id] = cache_temp_ptr->mu_mean_post;
			this->mu_prec_post[node_id] = cache_temp_ptr->mu_prec_post;
		}
		if (check_if_included(this->tree_ptr->leaf_node_ids, node_id)){
			if (stop_split(this->train_ids_new[node_id], control, train_data, cache)){
				this->logprior_new[node_id] = 0;
			}
			else{
				this->logprior_new[node_id] = std::log(compute_not_split_prob(this->tree_ptr, node_id, param));
			}
		}
		else{
			this->recompute_prob_split(train_data, param, control, cache, node_id);
		}
	}
}

void TreeMCMC::update_subtree(const IntVector& nodes_subtree) {
	UINT subtree_length = nodes_subtree.size();
	for (UINT i = 0; i < subtree_length; i++){
		UINT node_id = nodes_subtree[i];
		this->loglik[node_id] = this->loglik_new[node_id];
		this->logprior[node_id] = this->logprior_new[node_id];
		this->train_ids[node_id] = this->train_ids_new[node_id];
		this->sum_y[node_id] = this->sum_y_new[node_id];
		this->sum_y2[node_id] = this->sum_y2_new[node_id];
		this->n_points[node_id] = this->n_points_new[node_id];
		this->mu_mean_post[node_id] = this->mu_mean_post_new[node_id];
		this->mu_prec_post[node_id] = this->mu_prec_post_new[node_id];
	}
}

void TreeMCMC::recompute_prob_split(const Data& train_data, const Param& param, const Control& control,
  const Cache& cache, const UINT node_id) {
	const IntVector& train_ids = this->train_ids_new[node_id];
	if (stop_split(train_ids, control, train_data, cache)) {
		this->logprior_new[node_id] = -BART_DBL_MAX;
	}
	else {
		const SplitInfo&  n_info = this->node_info_new[node_id];
		UINT feat_id_chosen = n_info.feat_id_chosen;
		double split_chosen = n_info.split_chosen;
		UINT idx_split_global = n_info.idx_split_global;

		IntVector_Ptr feat_id_valid_ptr;
		DoubleVector_Ptr score_feat_ptr;
		map<UINT, DimensionInfo_Ptr> feat_split_info;
		bool split_not_supported;

		tie(feat_id_valid_ptr, score_feat_ptr, feat_split_info, split_not_supported) =
			find_valid_dimensions(train_data, cache, train_ids, control);

		if (!check_if_included(*feat_id_valid_ptr, feat_id_chosen)) {
			this->logprior_new[node_id] = -BART_DBL_MAX;
		}
		else {
			DoubleVector log_prob_feat;
			double tmp_sum = std::log(sum(*score_feat_ptr));
			for (double ele : *score_feat_ptr) {
				log_prob_feat.push_back(std::log(ele) - tmp_sum);
			}
			DimensionInfo_Ptr d_info = feat_split_info[feat_id_chosen];
			UINT idx_min = d_info->idx_min;
			UINT idx_max = d_info->idx_max;
			double x_min = d_info->x_min;
			double x_max = d_info->x_max;
			DoubleVector& feat_score_cumsum_prior_current = d_info->feat_score_cumsum_prior_current;
			if (split_chosen <= x_min || split_chosen >= x_max)
				this->logprior_new[node_id] = -BART_DBL_MAX;
			else {
				double z_prior = feat_score_cumsum_prior_current[idx_max] - feat_score_cumsum_prior_current[idx_min];
				DoubleVector prob_split_prior;
				double tmp;
				double offset = feat_score_cumsum_prior_current[idx_min];
				for (UINT i = idx_min; i <= idx_max; i++) {
					tmp = (feat_score_cumsum_prior_current[i] - offset) / z_prior;
					prob_split_prior.push_back(tmp);
				}
				UINT idx_split_chosen = idx_split_global - idx_min - 1;
				double logprior_nodeid_tau = std::log(prob_split_prior[idx_split_chosen]);
				double log_psplit = std::log(compute_split_prob(this->tree_ptr, node_id, param));
				this->logprior_new[node_id] = log_psplit + logprior_nodeid_tau + log_prob_feat[feat_id_chosen];
				// if (control.verbose >= 3) {
				// 	std::cout << "3 terms in recompute for node_id = " << node_id << "; "
				// 		<< log_psplit << "; " << logprior_nodeid_tau << "; " << log_prob_feat[feat_id_chosen] << std::endl;
				// 	std::cout << "feat_id = " << feat_id_chosen << ", idx_split_chosen = "
				// 		<< idx_split_chosen << ", split_chosen = " << split_chosen << std::endl;
				// 	std::cout << "log prob_split_prior = ";
				// 	for (double prob : prob_split_prior) {
				// 		std::cout << std::log(prob) << " ";
				// 	}
				// 	std::cout << std::endl;
				// }
			}
		}
	}
}

} // namesapce pgbart

namespace pgbart {

TreeMCMC_Ptr init_cgm_mcmc(const Data& train_data, const Control& control, const Param& param,
  const Cache& cache, const CacheTemp& cache_temp) {

  IntVector train_ids(range<UINT>(0, train_data.n_point));
  return make_shared<TreeMCMC>(train_ids, param, control, cache_temp);
}

tuple<TreeMCMC_Ptr, bool> run_cgm_mcmc_single_tree(TreeMCMC_Ptr tree_mcmc_ptr, const Control& control,
	const Data& train_data, const Param& param, const Cache& cache, const CacheTemp& cache_temp){

  const bool change = tree_mcmc_ptr->sample(train_data, control, param, cache);
  return make_tuple(tree_mcmc_ptr, change);
}

} // namespace pgbart

