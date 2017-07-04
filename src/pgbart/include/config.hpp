#ifndef PGBART_CONFIG_HPP
#define PGBART_CONFIG_HPP

#include <stdint.h>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <tuple>
#include <set>
#include <memory>
#include <sstream>

/**************************************
File name : config.hpp
Date : 2016-12-7
Struct List : SplitInfo
        DimensionInfo
        Param
        Cache
        CacheTemp
Typedef List :  unsigned int -- UINT;
        std::vector <bool> -- BoolVector;
        std::vector <double> -- DoubleVector;
        std::vector <UINT> -- IntVector;
        std::vector <std::string> -- StrVector;
        std::shared_ptr<IntVector> -- IntVector_Ptr;
        std::shared_ptr<DoubleVector> -- DoubleVector_Ptr;
        std::shared_ptr<SplitInfo> -- SplitInfo_Ptr;
        std::shared_ptr<DimensionInfo> -- DimensionInfo_Ptr;
        std::shared_ptr<Param> -- Param_Ptr;
        std::shared_ptr<Cache> -- Cache_Ptr;
        std::shared_ptr<CacheTemp> -- CacheTemp_Ptr;
        std::map<UINT, double> -- UNIT_double_map;
***************************************/

namespace pgbart {

struct SplitInfo;
struct DimensionInfo;
struct Param;
struct Cache;
struct CacheTemp;

using std::string;
using std::map;
using std::pair;
using std::vector;
using std::tuple;
using std::make_tuple;
using std::tie;
using std::make_pair;
using std::set;
using std::shared_ptr;
using std::make_shared;
using std::unique_ptr;

typedef unsigned int UINT;
typedef std::vector<bool> BoolVector;
typedef std::vector<double> DoubleVector;
typedef std::vector<UINT> IntVector;
typedef std::vector<std::string> StrVector;
// typedef std::vector<std::tuple> TupleVector;
typedef std::shared_ptr<IntVector> IntVector_Ptr;
typedef std::shared_ptr<DoubleVector> DoubleVector_Ptr;
typedef std::shared_ptr<SplitInfo> SplitInfo_Ptr;
typedef std::shared_ptr<DimensionInfo> DimensionInfo_Ptr;
typedef std::shared_ptr<Param> Param_Ptr;
typedef std::shared_ptr<Cache> Cache_Ptr;
typedef std::shared_ptr<CacheTemp> CacheTemp_Ptr;
typedef std::map<UINT, double> UNIT_double_map;

} // namespace pgbart

namespace pgbart {
struct MCMC_Status {
  double loglik;
  double logprior;
  double logprob;
  double mean_depth;
  double mean_num_leaves;
  double mean_num_nonleaves;
  double mean_change;
  double mse_train;
  double lambda_bart;
  double time_itr;

  MCMC_Status(){
    this->loglik = 0;
    this->logprior = 0;
    this->logprob = 0;
    this->mean_depth = 0;
    this->mean_num_leaves = 0;
    this->mean_num_nonleaves = 0;
    this->mean_change = 0;
    this->mse_train = 0;
    this->lambda_bart = 0;
    this->time_itr = 0;
  }
};

struct SplitInfo {
  UINT feat_id_chosen;
  double split_chosen;
  UINT idx_split_global;
  SplitInfo(UINT feat_id_chosen, double split_chosen, UINT idx_split_global) :
    feat_id_chosen(feat_id_chosen), split_chosen(split_chosen), idx_split_global(idx_split_global){}
  SplitInfo() : feat_id_chosen(0U), split_chosen(0), idx_split_global(0U){}
  string toString();

  bool operator==(const SplitInfo& other) const {
    if (this->feat_id_chosen != other.feat_id_chosen)
      return false;
    if (this->idx_split_global != other.idx_split_global)
      return false;
    if (this->split_chosen != other.split_chosen)
      return false;
    return true;
  }

  void operator=(const SplitInfo& other) {
    this->feat_id_chosen = other.feat_id_chosen;
    this->split_chosen = other.split_chosen;
    this->idx_split_global = other.idx_split_global;
  }

};

struct DimensionInfo {
  double x_min;
  double x_max;
  UINT idx_min;
  UINT idx_max;
  DoubleVector feat_score_cumsum_prior_current;

  DimensionInfo() : x_min(0), x_max(0), idx_min(0), idx_max(0), feat_score_cumsum_prior_current() {}

  DimensionInfo(double x_min, double x_max, UINT idx_min, UINT idx_max, DoubleVector feat_score_cumsum_prior_current) :
    x_min(x_min), x_max(x_max), idx_min(idx_min), idx_max(idx_max), feat_score_cumsum_prior_current(feat_score_cumsum_prior_current){}
  string toString();
};

struct Param {
  double alpha_bart;
  double alpha_split;
  double beta_bart;
  double beta_split;
  double k_bart;
  double lambda_bart;
  double log_lambda_bart;
  UINT m_bart;
  double mu_mean;
  double mu_prec;

  Param(double alpha_split, double beta_split) : alpha_split(alpha_split), beta_split(beta_split){}
  Param(){};
  string toString();
};

struct CacheTemp {
  bool isexisted;
  UINT n_points;
  double sum_y;
  double sum_y2;
  double loglik;
  double mu_mean_post;
  double mu_prec_post;

  UINT n_points_left;
  double sum_y_left;
  double sum_y2_left;
  double mu_mean_post_left;
  double mu_prec_post_left;
  double loglik_left;

  double sum_y_right;
  double sum_y2_right;
  UINT n_points_right;
  double mu_mean_post_right;
  double mu_prec_post_right;
  double loglik_right;

  CacheTemp(){
    isexisted = true;
  }
  string toString();
};

struct Cache {
  map<UINT, DoubleVector> feat_idx2midpoint;
  DoubleVector feat_k_log_prior;
  map<UINT, DoubleVector> feat_score_cumsum;
  map<UINT, DoubleVector> feat_score_cumsum_prior;
  map<UINT, map<double, UINT>> feat_val2idx;
  double half_log_2pi;
  double nn_prior_term;
  DoubleVector prob_feat;
  IntVector range_n_dim;
  IntVector range_n_dim_shuffle;
  double log_n_dim;
};

struct NodePair {
  UINT parent;
  UINT child;

  NodePair(): parent(0U), child(0U){}
  NodePair(UINT parent, UINT child): parent(parent), child(child){}
  bool operator==(const NodePair& other) const { 
    return this->parent == other.parent && this->child == other.child;
  }
  void operator=(const NodePair& other) {
    this->parent = other.parent;
    this->child = other.child;
  }
};

} // namespace pgbart

namespace pgbart {

#define BART_PI static_cast<double>(3.14159265358979323846)
#define BART_DBL_MAX 1.7976931348623158e+308 /* max value */

}

#endif
