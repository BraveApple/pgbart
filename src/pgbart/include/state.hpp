#ifndef PGBART_STATE_HPP
#define PGBART_STATE_HPP

#include <map>
#include <tuple>

#include "pgbart/include/tree.hpp"
#include "pgbart/include/config.hpp"
#include "pgbart/include/random.hpp"
#include "pgbart/include/math.hpp"
#include "pgbart/include/data.hpp"
#include "pgbart/include/control.hpp"
#include "pgbart/include/compare.hpp"

namespace pgbart {
class State {
public:
  shared_ptr<Tree> tree_ptr; // pointer to a tree
  map<UINT, bool> do_not_split; // whether a node can split
  map<UINT, double> sum_y; // sum of labels on a node
  map<UINT, double> sum_y2; // sum of square of labels on a node
  map<UINT, UINT> n_points; // number of data point on a node
  map<UINT, double> loglik; // loglik of a node
  map<UINT, double> mu_mean_post; //mean of the normal distribution of mu on a node, posterior probability 
  map<UINT, double> mu_prec_post; //stddev of the normal distribution of mu on a node, posterior probability

  map<UINT, IntVector> train_ids; // the id of data point on the node
  map<UINT, SplitInfo> node_info; // the split info of an internal node
  map<UINT, double> logprior;
  double loglik_current;
  map<UINT, tuple<double, double>> log_sis_ratio_d;

public:
  State(const IntVector& train_ids, const Param& param, const CacheTemp& cache_temp);
  State() { this->tree_ptr = make_shared<Tree>(); };

  /*
  return {"bool do_not_split_node_id", "SplitInfo split_info", "double log_sis_ratio", "double logprior_nodeid",
  "IntVector train_ids_left", "IntVector train_ids_right", "CacheTemp cache_temp"}
  */

  tuple<bool, SplitInfo_Ptr, double, double, IntVector_Ptr, IntVector_Ptr, CacheTemp_Ptr>
    prior_proposal(const Data& data_train, const UINT& node_id, const IntVector& train_ids, const Param& param, const Cache& cache,
    const Control& control);

  void update_left_right_statistics(const UINT& node_id, const double& logprior_nodeid, const SplitInfo& split_info,
    const CacheTemp& cache_temp, const Control& control, const IntVector& train_ids_left, const IntVector& train_ids_right,
    const Data& data_train, const Cache& cache, const Param& param);

  void remove_leaf_node_statistics(const UINT& node_id);

  /*
  return {"bool do_not_split_node_id", "SplitInfo split_info", "double logprior_nodeid"}
  */
  tuple<bool, SplitInfo_Ptr, double>
    sample_split_prior(const UINT& node_id, const Data& data_train, const Param& param, const Control& control, const Cache& cache);

  /*
  return {"IntVector feat_id_valid", "DoubleVector score_feat", "map<UINT, DimensionInfo> feat_split_info", "bool split_not_supported"}
  */
  tuple<IntVector_Ptr, DoubleVector_Ptr, map<UINT, DimensionInfo_Ptr>, bool>
    find_valid_dimensions(const Data& data_train, const Cache& cache, const IntVector& train_ids, const Control& control);
  
  double compute_logprior();

  void update_depth();

  double compute_loglik();

  double compute_logprob();

  void print_tree();

  void check_depth();

  IntVector* gen_rules_tree(const Data& data_train);

  DoubleVector* predict_real_val_fast(IntVector* leaf_id);

  void update_loglik_node(UINT node_id, const Data& data, const Param& param, const Cache& cache, const Control& control);

  void update_loglik_node_all(const Data& data, const Param& param, const Cache& cache, const Control& control);

};
}

namespace pgbart {
double compute_nn_loglik(double x, double mu, double prec, double log_const);

bool is_split_valid(const double& split_chosen, const double& x_min, const double& x_max);

bool stop_split(const IntVector& train_ids, const Control& control, const Data& data_train, const Cache& cache);

// python compute_log_pnosplit_childern(node_id, param)
double compute_not_split_children_log_prob(const shared_ptr<Tree> tree, UINT node_id, const Param& param);

// return {"IntVector feat_id_perm", "UINT n_feat", "DoubleVector log_prob_feat"}
tuple<IntVector_Ptr, UINT, DoubleVector_Ptr>
  score_features(const Control& control, const IntVector& feat_id_valid, const DoubleVector& score_feat, const bool& split_not_supported);

// return {"IntVector train_ids_left", "IntVector train_ids_right", "CacheTemp cache_temp"}
tuple<IntVector_Ptr, IntVector_Ptr, CacheTemp_Ptr>
  compute_left_right_statistics(const Data& data_train, const Param& param, const Control& control,
  const Cache& cache, const UINT& feat_id_chosen, const double& split_chosen, const IntVector& train_ids);

//return {"Param param", "Cache cache", "CacheTemp cache_temp"}
tuple<Param_Ptr, Cache_Ptr, CacheTemp_Ptr> precompute(const Data& data_train, const Control& control);

void update_cache_temp(CacheTemp& cache_temp, const Cache& cache, const Data& data, const Param& param, const Control& control);

bool no_valid_split_exists(const Data& data_train, const Cache& cache, const IntVector& train_ids);

// return {"double sum_y", "double sum_y2", "UINT n_points"}
tuple<double, double, UINT> get_reg_states(const DoubleVector& y);

extern "C"{
  double compute_gamma_param(double min_val, double alpha, double q, double init_val = -1.0);
}

double gammaln(double alpha);

double compute_gamma_loglik(double x, double alpha, double beta);

double compute_normal_loglik(double x, double mu, double prec);

double compute_split_prob(const shared_ptr<Tree> tree_ptr, const size_t& node_id, const Param& param);

double compute_not_split_prob(const shared_ptr<Tree> tree_ptr, const size_t& node_id, const Param& param);

void compute_normal_normalizer(const Param& param, const Cache& cache, CacheTemp& cache_temp, const string& str_node = "parent");

DimensionInfo_Ptr get_info_dimension(const Data& data_train, const Cache& cache, const IntVector& train_ids, const UINT& feat_id);
}

#endif