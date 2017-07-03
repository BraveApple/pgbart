#include <iostream>

#include "pgbart/include/state.hpp"
#include "external/include/mconf.h"
#include "external/include/gdtr.h"

using namespace pgbart::math;

namespace pgbart {
State::State(const IntVector& train_ids, const Param& param, const CacheTemp& cache_temp) {
  this->tree_ptr = make_shared<Tree>();
  if (cache_temp.isexisted) {
    this->tree_ptr->leaf_node_ids.push_back(this->tree_ptr->root_node_id);
    this->do_not_split[this->tree_ptr->root_node_id] = false; // insert a pair by key
    this->sum_y[this->tree_ptr->root_node_id] = cache_temp.sum_y;
    this->sum_y2[this->tree_ptr->root_node_id] = cache_temp.sum_y2;
    this->n_points[this->tree_ptr->root_node_id] = cache_temp.n_points;
    this->loglik[this->tree_ptr->root_node_id] = cache_temp.loglik;
    this->mu_mean_post[this->tree_ptr->root_node_id] = cache_temp.mu_mean_post;
    this->mu_prec_post[this->tree_ptr->root_node_id] = cache_temp.mu_prec_post;
    this->train_ids[this->tree_ptr->root_node_id] = train_ids;
    this->node_info.clear();
    this->logprior[this->tree_ptr->root_node_id] = std::log(compute_not_split_prob(this->tree_ptr, this->tree_ptr->root_node_id, param));
    this->loglik_current = this->loglik[this->tree_ptr->root_node_id] + 0.0;
  }
}

double State::compute_logprior() {
  double sum = 0.0;
  for (auto node_id : this->tree_ptr->leaf_node_ids)
    sum += this->logprior[node_id];
  for (auto node_id : this->tree_ptr->non_leaf_node_ids)
    sum += this->logprior[node_id];
  return sum;
}

double State::compute_loglik() {
  double sum = 0.0;
  for (auto node_id : this->tree_ptr->leaf_node_ids)
    sum += this->loglik[node_id];
  return sum;
}

/*
return {"bool do_not_split_node_id", "SplitInfo split_info", "double log_sis_ratio", "double logprior_nodeid",
"IntVector train_ids_left", "IntVector train_ids_right", "CacheTemp cache_temp"}
*/
tuple<bool, SplitInfo_Ptr, double, double, IntVector_Ptr, IntVector_Ptr, CacheTemp_Ptr>
State::prior_proposal(const Data& data_train, const UINT& node_id, const IntVector& train_ids, const Param& param, const Cache& cache,
const Control& control) {
  /*
  NOTE: a uniform prior over the features; see how feat_id_chosen is sampled below
  */
  bool do_not_split_node_id;
  SplitInfo_Ptr split_info_ptr(new SplitInfo());
  double log_sis_ratio;
  double logprior_nodeid;
  IntVector_Ptr train_ids_left_ptr;
  IntVector_Ptr train_ids_right_ptr;
  CacheTemp_Ptr cache_temp_ptr;

  double not_split_prob = compute_not_split_prob(this->tree_ptr, node_id, param); // See expression (4) in PGBART paper
  do_not_split_node_id = simulate_continuous_uniform_distribution(0.0, 1.0) <= not_split_prob;


  bool split_not_supported = false;

  if (!do_not_split_node_id) {
    // randomly switch an feature to split, if there is no avaliable split point, chose another feature
    const_cast<Cache&>(cache).range_n_dim_shuffle = shuffle(const_cast<Cache&>(cache).range_n_dim);
    for (auto feat_id_chosen_temp : cache.range_n_dim_shuffle) {
      DimensionInfo_Ptr demension_info_ptr = get_info_dimension(data_train, cache, train_ids, feat_id_chosen_temp);
      double x_min = demension_info_ptr->x_min;
      double x_max = demension_info_ptr->x_max;
      UINT idx_min = demension_info_ptr->idx_min;
      UINT idx_max = demension_info_ptr->idx_max;
      DoubleVector& feat_score_cumsum_prior_current = demension_info_ptr->feat_score_cumsum_prior_current;
      if (idx_min == idx_max) {
        // chose the next feature
        split_not_supported = true;
        continue;
      }
      double z_prior = feat_score_cumsum_prior_current[idx_max] - feat_score_cumsum_prior_current[idx_min];
      DoubleVector prob_split_prior = diff(at(feat_score_cumsum_prior_current, range(idx_min, idx_max + 1))- feat_score_cumsum_prior_current[idx_min]) / z_prior;
      UINT idx_split_chosen = sample_multinomial_distribution(prob_split_prior);
      UINT idx_split_global_temp = idx_split_chosen + idx_min + 1;
      double split_value_temp = const_cast<Cache&>(cache).feat_idx2midpoint[feat_id_chosen_temp][idx_split_global_temp];
      if (control.if_debug) {
        if (compare::compare_if_between(x_min, split_value_temp, x_max)) {
          std::cout << "the split_value = " << split_value_temp << " must be between x_min = " << x_min << " and x_max = " 
            << x_max << std::endl;
          exit(1);
        }
      }
      double logprior_nodeid_temp = 0.0;
      /*
      logprior_nodeid is not really needed to sample
      NOTE: using this incorrect value might result in invalid logprior, logprob values in BART
      */

      // compute the statistics for the two child node
      std::tie(train_ids_left_ptr, train_ids_right_ptr, cache_temp_ptr) = compute_left_right_statistics(
        data_train, param, control, cache, feat_id_chosen_temp, split_value_temp, train_ids);

      if (control.if_debug) {

        bool check = compare::compare_if_zero(cache_temp_ptr->sum_y_left + cache_temp_ptr->sum_y_right - this->sum_y[node_id]);
        check = check && compare::compare_if_zero(cache_temp_ptr->sum_y2_left + cache_temp_ptr->sum_y2_right - this->sum_y2[node_id]);
        check = check && compare::compare_if_zero(cache_temp_ptr->n_points_left + cache_temp_ptr->n_points_right - this->n_points[node_id]);

        if (!check) {
          std::cout << "sum_y = " << this->sum_y[node_id] << ", sum_y_left = " << cache_temp_ptr->sum_y_left << ", sum_y_right = "
            << cache_temp_ptr->sum_y_right << "sum_y_left + sum_y_right = " << cache_temp_ptr->sum_y_left + cache_temp_ptr->sum_y_right << std::endl;
          
          std::cout << "sum_y2 = " << this->sum_y2[node_id] << ", sum_y2_left = " << cache_temp_ptr->sum_y2_left << ", sum_y2_right = "
            << cache_temp_ptr->sum_y2_right << "sum_y2_left + sum_y2_right = " << cache_temp_ptr->sum_y2_left + cache_temp_ptr->sum_y2_right << std::endl;
          exit(1);
        }
      }

      if (control.verbose_level >= 2) {
        std::cout << "loglik (of all data points in parent) = " << this->loglik[node_id] << std::endl;
      }
      double log_sis_ratio_loglik = cache_temp_ptr->loglik_left + cache_temp_ptr->loglik_right - this->loglik[node_id];
      double log_sis_ratio_prior = 0.0;
      // contributions of feat_id and prob_split cancel out for precomputed proposals
      double log_sis_ratio_temp = log_sis_ratio_loglik + log_sis_ratio_prior;
      this->log_sis_ratio_d[node_id] = make_tuple(log_sis_ratio_loglik, log_sis_ratio_prior);

      if (control.verbose_level >= 2) {
        std::cout << "idx_split_chosen = " << idx_split_chosen << ", split_chosen = " << split_value_temp 
        << ", feat_id_chosen = " << feat_id_chosen_temp << std::endl;
      }
      split_not_supported = false;
      split_info_ptr->feat_id_chosen = feat_id_chosen_temp;
      split_info_ptr->split_chosen = split_value_temp;
      split_info_ptr->idx_split_global = idx_split_global_temp;
      logprior_nodeid = logprior_nodeid_temp;
      log_sis_ratio = log_sis_ratio_temp;
      break; // if you got this far, you deserve a break!
    }
  }
  if (split_not_supported)
    do_not_split_node_id = true;

  if (do_not_split_node_id) {
    // use defalut value
    split_info_ptr->feat_id_chosen = -1;
    split_info_ptr->split_chosen = 3.14;
    split_info_ptr->idx_split_global = -1;
    logprior_nodeid = 0.0;
    log_sis_ratio = 0.0;
    train_ids_left_ptr = IntVector_Ptr(new IntVector());
    train_ids_right_ptr = IntVector_Ptr(new IntVector());
    cache_temp_ptr = CacheTemp_Ptr(new CacheTemp());
    cache_temp_ptr->loglik_left = -DBL_MAX;
    cache_temp_ptr->loglik_right = -DBL_MAX;

  }
  return make_tuple(do_not_split_node_id, split_info_ptr, log_sis_ratio, logprior_nodeid,
    train_ids_left_ptr, train_ids_right_ptr, cache_temp_ptr);
}

void State::update_depth() {
	this->tree_ptr->updateTreeDepth();
}


void State::update_left_right_statistics(const UINT& node_id, const double& logprior_nodeid, const SplitInfo& split_info,
  const CacheTemp& cache_temp, const Control& control, const IntVector& train_ids_left, const IntVector& train_ids_right,
  const Data& data_train, const Cache& cache, const Param& param) {
  UINT left_node_id = this->tree_ptr->getLeftNodeID(node_id);
  UINT right_node_id = this->tree_ptr->getRightNodeID(node_id);
  this->logprior[node_id] = logprior_nodeid;
  this->node_info[node_id] = split_info;

  this->loglik[left_node_id] = cache_temp.loglik_left;
  this->loglik[right_node_id] = cache_temp.loglik_right;
  this->do_not_split[left_node_id] = stop_split(train_ids_left, control, data_train, cache);
  this->do_not_split[right_node_id] = stop_split(train_ids_right, control, data_train, cache);

  if (this->do_not_split[left_node_id])
    this->logprior[left_node_id] = 0.0;
  else
    this->logprior[left_node_id] = std::log(compute_not_split_prob(tree_ptr, left_node_id, param));

  if (this->do_not_split[right_node_id])
    this->logprior[right_node_id] = 0.0;
  else
    this->logprior[right_node_id] = std::log(compute_not_split_prob(tree_ptr, left_node_id, param));

  if (control.if_debug) {

    bool check = compare::compare_if(cache_temp.n_points_left, ">", 0U);
    check = check && compare::compare_if(cache_temp.n_points_right, ">", 0U);

    if (!check) {
      std::cout << "ERROR: \"n_points_left\" and \"n_points_right\" must be zero!" << std::endl;
      exit(1);
    }
  }

  this->train_ids[left_node_id] = train_ids_left;
  this->tree_ptr->leaf_node_ids.push_back(left_node_id);
  this->train_ids[right_node_id] = train_ids_right;
  this->tree_ptr->leaf_node_ids.push_back(right_node_id);
  this->sum_y[left_node_id] = cache_temp.sum_y_left;
  this->sum_y2[left_node_id] = cache_temp.sum_y2_left;
  this->n_points[left_node_id] = cache_temp.n_points_left;
  this->mu_mean_post[left_node_id] = cache_temp.mu_mean_post_left;
  this->mu_prec_post[left_node_id] = cache_temp.mu_prec_post_left;
  this->sum_y[right_node_id] = cache_temp.sum_y_right;
  this->sum_y2[right_node_id] = cache_temp.sum_y2_right;
  this->n_points[right_node_id] = cache_temp.n_points_right;
  this->mu_mean_post[right_node_id] = cache_temp.mu_mean_post_right;
  this->mu_prec_post[right_node_id] = cache_temp.mu_prec_post_right;

  if (!delete_element(this->tree_ptr->leaf_node_ids, node_id)) {
    std::cout << "ERROR: fail to delete \"node_id = " << node_id << "\"!" << std::endl;  
    exit(1);
  }
  this->tree_ptr->non_leaf_node_ids.push_back(node_id);
  this->tree_ptr->updateTreeDepth();
}

void State::remove_leaf_node_statistics(const UINT& node_id) {
  if (!delete_element(tree_ptr->leaf_node_ids, node_id)) {
    std::cout << "ERROR: fail to delete \"node_id = " << node_id << "\"!" << std::endl;  
  }
  else {
    math::delete_by_key(this->loglik, node_id);
    math::delete_by_key(this->train_ids, node_id);
    math::delete_by_key(this->logprior, node_id);
    math::delete_by_key(this->sum_y, node_id);
    math::delete_by_key(this->sum_y2, node_id);
    math::delete_by_key(this->n_points, node_id);
    math::delete_by_key(this->mu_mean_post, node_id);
    math::delete_by_key(this->mu_prec_post, node_id);
  }
}

/*
return {"IntVector feat_id_valid", "DoubleVector score_feat", "map<UINT, DimensionInfo> feat_split_info", "bool split_not_supported"}
*/
tuple<IntVector_Ptr, DoubleVector_Ptr, map<UINT, DimensionInfo_Ptr>, bool>
  State::find_valid_dimensions(const Data& data_train, const Cache& cache, const IntVector& train_ids, const Control& control) {
  IntVector_Ptr feat_id_valid_ptr(new IntVector());
  DoubleVector_Ptr score_feat_ptr(new DoubleVector());
  map<UINT, DimensionInfo_Ptr> feat_split_info;
  bool split_not_supported;

  *score_feat_ptr = cache.prob_feat;
  bool first_time = true;
  if (control.verbose_level >= 3)
    std::cout << "original score_feat = " << toString(*score_feat_ptr) << std::endl;
  for (auto feat_id : cache.range_n_dim) {
    DimensionInfo_Ptr demension_info_ptr = get_info_dimension(data_train, cache, train_ids, feat_id);
    demension_info_ptr->feat_score_cumsum_prior_current = const_cast<Cache&>(cache).feat_score_cumsum_prior[feat_id];
    if (control.verbose_level >= 3) {
      std::cout << "x_min = " << demension_info_ptr->x_min << ", x_max = " << demension_info_ptr->x_max 
        << ", idx_min = " << demension_info_ptr->idx_min << ", idx_max = " << demension_info_ptr->idx_min << std::endl;
    }

    if (demension_info_ptr->idx_min == demension_info_ptr->idx_max) {
      if (first_time) { // lazy copy
        *score_feat_ptr = cache.prob_feat;
        first_time = false;
      }
      (*score_feat_ptr)[feat_id] = 0.0;
    }
    else {
      feat_split_info.insert(make_pair(feat_id, demension_info_ptr));
    }
  }
  //IntVector feat_id_valid;
  for (auto feat_id : cache.range_n_dim) {
    if ((*score_feat_ptr)[feat_id] > 0.0)
      feat_id_valid_ptr->push_back(feat_id);
  }

  split_not_supported = feat_id_valid_ptr->empty();

  if (control.verbose_level >= 3) {
    Matrix<double>& x = const_cast<Data&>(data_train).x;
    std::cout << "in find_valid_dimensions now!" << std::endl;
    std::cout << "training data in current node = \n" <<  x(train_ids, ":").toString() << std::endl;
    std::cout << "score_feat = " << toString(*score_feat_ptr) << std::endl;
    std::cout << "feat_id_valid = " << toString(*feat_id_valid_ptr) << std::endl;
  }

  return make_tuple(feat_id_valid_ptr, score_feat_ptr, feat_split_info, split_not_supported);
}

/*
return {"bool do_not_split_node_id", "SplitInfo split_info", "double logprior_nodeid"}
*/
tuple<bool, SplitInfo_Ptr, double>
  State::sample_split_prior(const UINT& node_id, const Data& data_train, const Param& param, const Control& control, const Cache& cache) {
  bool do_not_split_node_id;
  SplitInfo_Ptr split_info_ptr(new SplitInfo());
  double logprior_nodeid;


  IntVector& train_ids = this->train_ids[node_id];
  //UINT n_train_ids = train_ids.size();
  double log_prob_split = std::log(compute_split_prob(this->tree_ptr, node_id, param));
  //double prob_not_split = compute_not_split_prob(this->tree_ptr, node_id, param);

  IntVector_Ptr feat_id_valid_ptr;
  DoubleVector_Ptr score_feat_ptr;
  map<UINT, DimensionInfo_Ptr> feat_split_info;
  bool split_not_supported;

  tie(feat_id_valid_ptr, score_feat_ptr, feat_split_info, split_not_supported) = 
    find_valid_dimensions(data_train, cache, train_ids, control);

  if (split_not_supported) {
    do_not_split_node_id = true;
    split_info_ptr->feat_id_chosen = -1;
    split_info_ptr->split_chosen = 3.14; // i like pi :)
    split_info_ptr->idx_split_global = -1;
    logprior_nodeid = 0.0;
  }
  else {
    do_not_split_node_id = false;
    IntVector_Ptr feat_id_perm_ptr;
    UINT n_feat;
    DoubleVector_Ptr log_prob_feat_ptr;

    tie(feat_id_perm_ptr, n_feat, log_prob_feat_ptr) =
      score_features(control, *feat_id_valid_ptr, *score_feat_ptr, split_not_supported);

    UINT feat_id_chosen = sample_multinomial_scores(*score_feat_ptr);
    split_info_ptr->feat_id_chosen = feat_id_chosen;
    DimensionInfo_Ptr demension_info_ptr = at(feat_split_info, feat_id_chosen);
    UINT idx_min = demension_info_ptr->idx_min;
    UINT idx_max = demension_info_ptr->idx_max;
    DoubleVector& feat_score_cumsum_prior_current = demension_info_ptr->feat_score_cumsum_prior_current;
    double z_prior = feat_score_cumsum_prior_current[idx_max] - feat_score_cumsum_prior_current[idx_min];
    DoubleVector prob_split_prior = diff(at(feat_score_cumsum_prior_current, range(idx_min, idx_max))
      - feat_score_cumsum_prior_current[idx_min]) / z_prior;
    UINT idx_split_chosen = sample_multinomial_distribution(prob_split_prior);
    UINT idx_split_global = idx_split_chosen + idx_min + 1;
    split_info_ptr->idx_split_global = idx_split_global;
    double split_chosen = const_cast<Cache&>(cache).feat_idx2midpoint[feat_id_chosen][idx_split_global];
    split_info_ptr->split_chosen = split_chosen;
    double x_min = demension_info_ptr->x_min;
    double x_max = demension_info_ptr->x_max;
    if (control.if_debug && !is_split_valid(split_chosen, x_min, x_max)) {
      std::cout << "ERROR: the split_chosen = " << split_chosen << " must be between x_min = " << x_min << " and x_max = " << x_max << std::endl;
      exit(1);
    }
    double logprior_nodeid_tau = std::log(prob_split_prior[idx_split_chosen]);
    logprior_nodeid = log_prob_split + logprior_nodeid_tau + (*log_prob_feat_ptr)[feat_id_chosen];

    if (control.verbose_level >= 2) {
      std::cout << "idx_split_chosen = " << idx_split_chosen << ", split_chosen = " << split_chosen 
        << ", feat_id_chosen = " << feat_id_chosen << std::endl;
    }
    if (control.verbose_level >= 3) {
      std::cout << "3 terms in sample_split_prior for node_id = " << node_id << "{ " << log_prob_split << ", "
        << logprior_nodeid_tau << ", " << (*log_prob_feat_ptr)[feat_id_chosen] << std::endl;
      std::cout << "log prob_split_prior = \n" << toString(log(prob_split_prior)) << std::endl;
    }
  }

  return make_tuple(do_not_split_node_id, split_info_ptr, logprior_nodeid);
}

void State::print_tree() {
  std::cout << "leaf nodes are \n" << toString(this->tree_ptr->leaf_node_ids) << std::endl;
  std::cout << "non-leaf nodes are \n" << toString(this->tree_ptr->non_leaf_node_ids) << std::endl;
  std::cout << "logprior = \n" <<  toString(this->logprior) << std::endl;
  std::cout << "loglik = \n" << toString(this->loglik) << std::endl;
  std::cout << "sum(logprior) = " << this->compute_logprior() << ", sum(loglik) = " << this->compute_loglik() << std::endl;
  std::cout << "leaf nodes are \n" << toString(this->tree_ptr->leaf_node_ids) << std::endl;
  std::cout << "node_id\t\tdepth\t\tfeat_id\t\tsplit_point" << std::endl;
  for (size_t node_id = 0; node_id < this->node_info.size(); node_id++) {
    SplitInfo& split_info = this->node_info[node_id];
    UINT feat_id = split_info.feat_id_chosen;
    double split = split_info.split_chosen;
    //UINT idx_split_global = split_info.idx_split_global;
    std::cout << node_id << "\t\t" << this->tree_ptr->getNodeDepth(node_id) << "\t\t" << feat_id << "\t\t" << split << std::endl;
  }
}

IntVector* State::gen_rules_tree(const Data& data_train){
  IntVector* leaf_id = new IntVector(data_train.x.n_row, 0);
  UINT row = data_train.x.n_row;
  UINT column = data_train.x.n_column;
  for (UINT node_id : this->tree_ptr->leaf_node_ids){
    BoolVector condition_vector(data_train.x.n_row, true);
    if (node_id == 0)
      break;
    UINT nid = node_id;
    while (nid != 0){
      //UINT pid = get_parent_id(nid);
      UINT pid = this->tree_ptr->getParentNodeID(nid);
      UINT feature_id = this->node_info[pid].feat_id_chosen;
      double split_value = this->node_info[pid].split_chosen;

      if (nid == (2 * pid + 1)){
        for (UINT i = 0; i < row; i++){
          if (data_train.x.elements[i * column + feature_id] > split_value)
            condition_vector[i] = condition_vector[i] & false;
        }
      }
      else{
        for (UINT i = 0; i < row; i++){
          if (data_train.x.elements[i * column + feature_id] <= split_value)
            condition_vector[i] = condition_vector[i] & false;
        }
      }
      nid = pid;
    }
    for (UINT j = 0; j < row; j++){
      if (condition_vector[j])
        leaf_id->at(j) = node_id;
    }
  }
  return leaf_id;
}

DoubleVector* State::predict_real_val_fast(IntVector* leaf_id){
  DoubleVector* pred_val = new DoubleVector(leaf_id->size());
  UINT length = leaf_id->size();
  for (UINT i = 0; i < length; i++){
    pred_val->at(i) = (this->tree_ptr->pred_val_n[leaf_id->at(i)]);
  }
  return pred_val;
}

void State::check_depth() {
  UINT max_leaf = max(this->tree_ptr->leaf_node_ids);
  UINT depth_max_leaf = this->tree_ptr->getNodeDepth(max_leaf);

  if (this->tree_ptr->getTreeDepth() != depth_max_leaf) {
    if (max_leaf != 0) {
      std::cout << "Error in check_depth: tree_depth = " << this->tree_ptr->getTreeDepth() << ", max_leaf = "
        << max_leaf << ", depth(max_leaf) = " << depth_max_leaf << std::endl;
      exit(1);
    }
  }
}

double State::compute_logprob() {
  return this->compute_loglik() + this->compute_logprior();
}

void State::update_loglik_node(UINT node_id, const Data& data, const Param& param, const Cache& cache, const Control& control) {
  double sum_y, sum_y2; UINT n_points;
  tie(sum_y, sum_y2, n_points) = get_reg_states(at(data.y_residual, train_ids[node_id]));
  this->sum_y[node_id] = sum_y;
  this->sum_y2[node_id] = sum_y2;

  double mu_prec_post = param.lambda_bart * n_points + param.mu_prec;
  this->mu_prec_post[node_id] = mu_prec_post;
  double mu_mean_post = (param.mu_prec * param.mu_mean + param.lambda_bart * sum_y) / mu_prec_post;
  this->mu_mean_post[node_id] = mu_mean_post;
  this->loglik[node_id] = cache.nn_prior_term - n_points * cache.half_log_2pi + 0.5 * (n_points * param.log_lambda_bart - std::log(mu_prec_post)
    + mu_prec_post * mu_mean_post * mu_mean_post - param.lambda_bart * sum_y2);
}

// update loglik for each node
void State::update_loglik_node_all(const Data& data, const Param& param, const Cache& cache, const Control& control) {
  for (auto pair : loglik) {
    UINT node_id = pair.first;
    this->update_loglik_node(node_id, data, param, cache, control);
  }
}

} // namespace pgbart

// *********************************************************************************************************

// *********************************************************************************************************

namespace pgbart {

double compute_nn_loglik(double x, double mu, double prec, double log_const) {
  return -0.5 * prec * (x - mu) * (x - mu) + log_const;
}

bool is_split_valid(const double& split_chosen, const double& x_min, const double& x_max) {
  return compare::compare_if_between(x_min, split_chosen, x_max);
}

bool stop_split(const IntVector& train_ids, const Control& control, const Data& data_train, const Cache& cache) {
  if (train_ids.size() <= control.min_size)
    return true;
  else
    return no_valid_split_exists(data_train, cache, train_ids);
}

// python compute_log_pnosplit_childern(node_id, param)
double compute_not_split_children_log_prob(const shared_ptr<Tree> tree_ptr, UINT node_id, const Param& param) {
  UINT left_node_id = tree_ptr->getLeftNodeID(node_id);
  UINT right_node_id = tree_ptr->getRightNodeID(node_id);
  double temp = std::log(1 - param.alpha_split * std::pow(1 + tree_ptr->getNodeDepth(left_node_id), -param.beta_split))
    + std::log(1 - param.alpha_split * std::pow(1 + tree_ptr->getNodeDepth(right_node_id), -param.beta_split));
  return temp;
}

// return {"IntVector feat_id_perm", "UINT n_feat", "DoubleVector log_prob_feat"}
tuple<IntVector_Ptr, UINT, DoubleVector_Ptr>
  score_features(const Control& control, const IntVector& feat_id_valid, const DoubleVector& score_feat, const bool& split_not_supported) {
  IntVector_Ptr feat_id_perm_ptr(new IntVector());
  UINT n_feat;
  DoubleVector_Ptr log_prob_feat_ptr(new DoubleVector());

  *feat_id_perm_ptr = feat_id_valid;
  n_feat = feat_id_perm_ptr->size();
  if (split_not_supported)
    *log_prob_feat_ptr = ones<double>(score_feat.size()) * static_cast<double>(NAN);
  else {
    *log_prob_feat_ptr = log(score_feat) - std::log(sum(score_feat));
    if (control.if_debug && !feat_id_perm_ptr->empty()) {
      if (absolute(log_sum_exp(*log_prob_feat_ptr)) >= 1e-12) {
        std::cout << "feat_id_perm = " << toString(*feat_id_perm_ptr) << std::endl;
        std::cout << "score_feat = " << toString(score_feat) << std::endl;
        std::cout << "log_sum_exp(log_prob_feat) = " << log_sum_exp(*log_prob_feat_ptr) << " (needs to be 0)" << std::endl;
        exit(1);
      }
    }
  }

  return make_tuple(feat_id_perm_ptr, n_feat, log_prob_feat_ptr);
}

// return {"IntVector train_ids_left", "IntVector train_ids_right", "CacheTemp cache_temp"}

tuple<IntVector_Ptr, IntVector_Ptr, CacheTemp_Ptr>
  compute_left_right_statistics(const Data& data_train, const Param& param, const Control& control,
  const Cache& cache, const UINT& feat_id_chosen, const double& split_chosen, const IntVector& train_ids) {
  IntVector_Ptr train_ids_left_ptr(new IntVector());
  IntVector_Ptr train_ids_right_ptr(new IntVector());
  CacheTemp_Ptr cache_temp_ptr(new CacheTemp());

  Matrix<double>& x = const_cast<Data&>(data_train).x;
  DoubleVector& y = const_cast<Data&>(data_train).y_residual;

  *train_ids_left_ptr = at(train_ids, choose_ids(x(train_ids, feat_id_chosen), split_chosen, "<="));
  *train_ids_right_ptr = at(train_ids, choose_ids(x(train_ids, feat_id_chosen), split_chosen, ">"));

  double sum_y_left = sum(at(y, *train_ids_left_ptr));
  double sum_y2_left = sum2(at(y, *train_ids_left_ptr));
  UINT n_points_left = train_ids_left_ptr->size();

  cache_temp_ptr->sum_y_left = sum_y_left;
  cache_temp_ptr->sum_y2_left = sum_y2_left;
  cache_temp_ptr->n_points_left = n_points_left;
  compute_normal_normalizer(param, cache, *cache_temp_ptr, "left");

  double sum_y_right = sum(at(y, *train_ids_right_ptr));
  double sum_y2_right = sum2(at(y, *train_ids_right_ptr));
  UINT n_points_right = train_ids_right_ptr->size();
  cache_temp_ptr->sum_y_right = sum_y_right;
  cache_temp_ptr->sum_y2_right = sum_y2_right;
  cache_temp_ptr->n_points_right = n_points_right;
  compute_normal_normalizer(param, cache, *cache_temp_ptr, "right");

  if (control.verbose_level >= 2) {
    std::cout << "feat_id_chosen = " << feat_id_chosen << ", split_chosen = " << split_chosen << std::endl;
    std::cout << "y (left) = " << toString(at(y, *train_ids_left_ptr)) << std::endl;
    std::cout << "y (right) = " << toString(at(y, *train_ids_right_ptr)) << std::endl;
    std::cout << "loglik (left) = " << cache_temp_ptr->loglik_left << ", loglik (right) = " << cache_temp_ptr->loglik_right << std::endl;
  }

  return make_tuple(train_ids_left_ptr, train_ids_right_ptr, cache_temp_ptr);
}

//return {"Param param", "Cache cache", "CacheTemp cache_temp"}
tuple<Param_Ptr, Cache_Ptr, CacheTemp_Ptr> precompute(const Data& data_train, const Control& control) {
  Param_Ptr param_ptr(new Param());
  Cache_Ptr cache_ptr(new Cache());
  CacheTemp_Ptr cache_temp_ptr(new CacheTemp());

  param_ptr->alpha_split = control.alpha_split;
  param_ptr->beta_split = control.beta_split;

  // BART prior
  param_ptr->m_bart = control.m_bart;
  param_ptr->k_bart = control.k_bart;
  /*
  See section 2.2.3 in page 271 of BART paper for how these parameters are set
  mu_mean, mu_prec are priors over gaussian in leaf node of each individual tree
  Basic idea is to choose mu_mean and mu_prec such that the interval [y_min, y_max] (see below)
  contains E[Y|X=x] with a specified probability (decided by k_bart: prob=0.95 if k_bart=2)
  y_min = m_bart * mu_mean - k_bart * sqrt(m_bart / mu_prec)
  y_max = m_bart * mu_mean + k_bart * sqrt(m_bart / mu_prec)
  */
  double y_max = max(data_train.y_residual);
  double y_min = min(data_train.y_residual);
  double y_mean = mean(data_train.y_residual);
  double y_diff = y_max - y_min;
  param_ptr->mu_mean = (y_min + 0.5 * y_diff) / param_ptr->m_bart;
  param_ptr->mu_prec = param_ptr->m_bart * std::pow(2 * param_ptr->k_bart / y_diff, 2.0);
  /*
  See section 2.2.4 of BART paper for how these parameters are set
  */

  double var_unconditional = variance(data_train.y_residual, y_mean);
  double prec_unconditional = 1.0 / var_unconditional;

  std::cout << "unconditional variance = " << var_unconditional << ", prec = " << prec_unconditional << std::endl;

  double prec = 0.0;

  if (control.variance_type == "leastsquares") {
    double ls_sum_squared_residuals = 0.0;
    double ls_var = ls_sum_squared_residuals / (data_train.n_point - 1);
    prec = 1.0 / ls_var;
    std::cout << "least squares variance = " << ls_var << ", prec = " << prec << std::endl;
    if (prec < prec_unconditional)
      std::cout << "least squares variance seems higher than unconditional ... something is weird!" << std::endl;
  }
  else if(control.variance_type == "unconditional") {
    std::cout << "WARNING: lambda_bart initialized to unconditional precision!" << std::endl;
    prec = prec_unconditional;
  }
  else {
    std::cout << "control.variance_type must be \"leastsquares\" or \"unconditional\"!" << std::endl;
    exit(1);
  }

  param_ptr->alpha_bart = control.alpha_bart;
  param_ptr->beta_bart = compute_gamma_param(prec, param_ptr->alpha_bart, control.q_bart);
  /*
  ensures that 1-gamcdf(prec; shape=alpha_bart, rate=beta_bart) /approx control.q_bart
  i.e. all values of precision are higher than the unconditional variance of Y
  param.lambda_bart = param.alpha_bart / param.beta_bart      #FIXME: better init? check sensitivity
  */

  if (control.variance_type == "leastsquares")
    param_ptr->lambda_bart = prec;
  else if(control.variance_type == "unconditional")
    param_ptr->lambda_bart = prec * 2;   // unconditional precision might be too pessimistic
  else {
    std::cout << "control.variance_type must be \"leastsquares\" or \"unconditional\"!" << std::endl;
    exit(1);
  }

  std::cout << "unconditional precision = " << prec_unconditional << ", initial lambda_bart = " << param_ptr->lambda_bart << std::endl;
  const_cast<Control&>(control).lambda_bart = param_ptr->lambda_bart;
  param_ptr->log_lambda_bart = std::log(param_ptr->lambda_bart);

  cache_temp_ptr->n_points = data_train.n_point;
  cache_temp_ptr->sum_y = sum(data_train.y_residual) / control.m_bart;
  cache_temp_ptr->sum_y2 = sum2(data_train.y_residual) / (control.m_bart * control.m_bart);
  cache_temp_ptr->isexisted = true;

  // pre-compute stuff
  //Cache cache;
  cache_ptr->nn_prior_term = 0.5 * std::log(param_ptr->mu_prec) - 0.5 * param_ptr->mu_prec * param_ptr->mu_mean * param_ptr->mu_mean;
  cache_ptr->half_log_2pi = 0.5 * std::log(2 * PI);
  // compute cache.nn_prior_term and cache.half_lod_2pi at first, otherwise fail to implement compute_normal_normalizer
  compute_normal_normalizer(*param_ptr, *cache_ptr, *cache_temp_ptr, "parent");

  cache_ptr->range_n_dim = range(0U, data_train.n_feature);
  cache_ptr->range_n_dim_shuffle = shuffle(range(0U, data_train.n_feature));
  cache_ptr->log_n_dim = std::log(data_train.n_feature);

  // log prior of k
  cache_ptr->feat_k_log_prior = ones<double>(data_train.n_feature) * (-1 * std::log(data_train.n_feature));

  // find all potential split feature and split value
  for (auto feat_id : cache_ptr->range_n_dim) {
    Matrix<double>& x = const_cast<Data&>(data_train).x;
    DoubleVector x_tmp = x(":", feat_id);
    IntVector idx_sort = argsort(x_tmp);
    DoubleVector x_tmp_sort(x_tmp.size(), 0);
    for (UINT i = 0; i < x_tmp.size(); i++)
      x_tmp_sort[i] = x_tmp[idx_sort[i]];
    DoubleVector feat_unique_values = math::unique(x_tmp_sort);
    cache_ptr->feat_val2idx[feat_id].clear();
    UINT n_unique = feat_unique_values.size();
    // even min value may be looked up
    for (UINT i = 0; i < n_unique; i++)
      cache_ptr->feat_val2idx[feat_id].insert(pair<double, UINT>(feat_unique_values[i], i));
    // first "interval" has width 0 since points to the left of that point are chosen with prob 0
    cache_ptr->feat_idx2midpoint[feat_id] = math::zeros<double>(n_unique);
    for (UINT i = 1; i < n_unique; i++)
      cache_ptr->feat_idx2midpoint[feat_id][i] = (feat_unique_values[i - 1] + feat_unique_values[i]) / 2.0;
    // each interval is represented by its midpoint
    DoubleVector diff_feat_unique_values = math::diff(feat_unique_values);
    DoubleVector log_diff_feat_unique_values_norm = math::log(diff_feat_unique_values) - std::log(feat_unique_values[n_unique - 1] - feat_unique_values[0]);
    DoubleVector feat_score_prior_tmp = math::zeros<double>(n_unique);
    math::replace(feat_score_prior_tmp, diff_feat_unique_values, 1);
    cache_ptr->feat_score_cumsum_prior[feat_id] = math::cumsum(feat_score_prior_tmp);

    if (control.if_debug) {
      /*
      print 'check if all these numbers are the same:'
      print n_unique, len(feat_score_cumsum_prior[feat_id])
      */
      if (n_unique != cache_ptr->feat_score_cumsum_prior[feat_id].size()) {
        std::cout << "ERROR: both of the size must be equal!" << std::endl;
        exit(1);
      }
    }
    if (control.verbose_level >= 3) {
      std::cout << "x (sorted) = " << toString(at(x_tmp, idx_sort)) << std::endl;
      std::cout << "y (corresponding to sorted x) = " << toString(at(data_train.y_residual, idx_sort)) << std::endl;
    }
  }
  cache_ptr->feat_score_cumsum = cache_ptr->feat_score_cumsum_prior;
  // use prob_feat instead of score_feat here; else need to pass sum of scores to log_sis_ratio
  cache_ptr->prob_feat = exp(cache_ptr->feat_k_log_prior);

  return make_tuple(param_ptr, cache_ptr, cache_temp_ptr);
}

void update_cache_temp(CacheTemp& cache_temp, const Cache& cache, const Data& data, const Param& param, const Control& control) {
  cache_temp.sum_y = sum(data.y_residual);
  cache_temp.sum_y2 = sum2(data.y_residual);
  compute_normal_normalizer(param, cache, cache_temp, "parent");
}

bool no_valid_split_exists(const Data& data_train, const Cache& cache, const IntVector& train_ids) {
  // faster way to check for existence of valid split than find_valid_dimensions
  bool op = true;
  for (auto feat_id : cache.range_n_dim_shuffle) {
    DimensionInfo_Ptr demension_info_ptr = get_info_dimension(data_train, cache, train_ids, feat_id);

    if (demension_info_ptr->idx_min != demension_info_ptr->idx_max) {
      op = false;
      break;
    }
  }
  return op;
}

// return {"double sum_y", "double sum_y2", "UINT n_points"}
tuple<double, double, UINT> get_reg_states(const DoubleVector& y) { 
  // y is a list of numbers, get_reg_stats(y) returns required for computing regression likelihood
  double sum_y = sum(y);
  double sum_y2 = sum2(y);
  UINT n_points = y.size();
  return make_tuple(sum_y, sum_y2, n_points);
}

extern "C"{
  double compute_gamma_param(double min_val, double alpha, double q, double init_val) {
    if (init_val < 0) {
      init_val = alpha / 3.0 / min_val;
    }
    double f1, f2 = 0;
    double x1 = init_val;
    double x2 = init_val;

    do {
      if (x2 < 0){
        x1 = 0;
      }
      else {
        x1 = x2;
      }
      f1 = gdtrc(x1, alpha, min_val) - q;
      f2 = (gdtrc(x1 + 0.0005, alpha, min_val) - gdtrc(x1, alpha, min_val)) / 0.0005;
      x2 = x1 - f1 / f2;
    } while (abs(x1 - x2) > 1.49012e-8);
    try{
      if (abs(gdtrc(x1, alpha, min_val) - q) > 1e-3) {
        throw x1;
      }
    }
    catch (double) {
      std::cout << "Failed to obtain the right solution: beta_init = " << init_val << ", q = " << q 
        << ", gdtrc(solution, alpha, min_val) = " << gdtrc(x1, alpha, min_val) << std::endl; 
      std::cout << "Trying a new initial value for beta!" << std::endl;
      double new_init = max(0.001, init_val * 0.9);
      x1 = compute_gamma_param(min_val, alpha, q, new_init);
    }
    return x1;
  }
}

double gammaln(double alpha) {
  return std::log(std::tgamma(alpha));
}

double compute_gamma_loglik(double x, double alpha, double beta) {
  double log_const = alpha * std::log(beta) - gammaln(alpha);
  double log_px = log_const + (alpha - 1) * std::log(x) - beta * x;
  return log_px;
}

double compute_normal_loglik(double x, double mu, double prec) {
  return 0.5 * (std::log(prec) - std::log(2 * PI) - prec * (x - mu) * (x - mu));
}

double compute_split_prob(const shared_ptr<Tree> tree, const size_t& node_id, const Param& param) {
  UINT depth = tree->getNodeDepth(node_id);
  return param.alpha_split * pow(1 + depth, -1 * param.beta_split);
}

double compute_not_split_prob(const shared_ptr<Tree> tree_ptr, const size_t& node_id, const Param& param) {
  return 1.0 - compute_split_prob(tree_ptr, node_id, param);
}

void compute_normal_normalizer(const Param& param, const Cache& cache, CacheTemp& cache_temp, const string& str_node) {
  if (str_node == "parent") {
    cache_temp.mu_prec_post = param.lambda_bart * cache_temp.n_points + param.mu_prec;
    cache_temp.mu_mean_post = (param.mu_prec * param.mu_mean + param.lambda_bart * cache_temp.sum_y) / cache_temp.mu_prec_post;
    cache_temp.loglik = cache.nn_prior_term - cache_temp.n_points * cache.half_log_2pi + 0.5 * (cache_temp.n_points * param.log_lambda_bart
      - std::log(cache_temp.mu_prec_post) + cache_temp.mu_prec_post * cache_temp.mu_mean_post * cache_temp.mu_mean_post - param.lambda_bart * cache_temp.sum_y2);
  }
  else if (str_node == "left") {
    cache_temp.mu_prec_post_left = param.lambda_bart * cache_temp.n_points_left + param.mu_prec;
    cache_temp.mu_mean_post_left = (param.mu_prec * param.mu_mean + param.lambda_bart * cache_temp.sum_y_left) / cache_temp.mu_prec_post_left;
    cache_temp.loglik_left = cache.nn_prior_term - cache_temp.n_points_left * cache.half_log_2pi + 0.5 * (cache_temp.n_points_left * param.log_lambda_bart
      - std::log(cache_temp.mu_prec_post_left) + cache_temp.mu_prec_post_left * cache_temp.mu_mean_post_left * cache_temp.mu_mean_post_left
      - param.lambda_bart * cache_temp.sum_y2_left);
  }
  else if (str_node == "right") {
    cache_temp.mu_prec_post_right = param.lambda_bart * cache_temp.n_points_right + param.mu_prec;
    cache_temp.mu_mean_post_right = (param.mu_prec * param.mu_mean + param.lambda_bart * cache_temp.sum_y_right) / cache_temp.mu_prec_post_right;
    cache_temp.loglik_right = cache.nn_prior_term - cache_temp.n_points_right * cache.half_log_2pi + 0.5 * (cache_temp.n_points_right * param.log_lambda_bart
      - std::log(cache_temp.mu_prec_post_right) + cache_temp.mu_prec_post_right * cache_temp.mu_mean_post_right * cache_temp.mu_mean_post_right
      - param.lambda_bart * cache_temp.sum_y2_right);
  }
  else {
    std::cout << "the 3rd parameter must be one of {\"left\", \"right\", \"parent\"}!" << std::endl;
    exit(1);
  }
}

DimensionInfo_Ptr get_info_dimension(const Data& data_train, const Cache& cache, const IntVector& train_ids, const UINT& feat_id) {
  
  DimensionInfo_Ptr dimension_info_ptr(new DimensionInfo());
  Matrix<double>& x = const_cast<Data&>(data_train).x;
  dimension_info_ptr->x_min = min(x(train_ids, feat_id));
  dimension_info_ptr->x_max = max(x(train_ids, feat_id));
  dimension_info_ptr->idx_min = at(const_cast<Cache&>(cache).feat_val2idx[feat_id], dimension_info_ptr->x_min);
  dimension_info_ptr->idx_max = at(const_cast<Cache&>(cache).feat_val2idx[feat_id], dimension_info_ptr->x_max);
  dimension_info_ptr->feat_score_cumsum_prior_current = const_cast<Cache&>(cache).feat_score_cumsum_prior[feat_id];

  return dimension_info_ptr;
}

} // namespace pgbart
