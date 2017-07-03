#ifndef PGBART_BART_HPP
#define PGBART_BART_HPP

#include "pgbart/include/config.hpp"
#include "pgbart/include/control.hpp"
#include "pgbart/include/data.hpp"
#include "pgbart/include/mcmc.hpp"

namespace pgbart {
class Bart {
public:

  vector<Particle_Ptr> p_particles; // ptr to the particles
  vector<Pmcmc_Ptr> pmcmc_objects;  // object to mcmc, note that the trees and pmcmc_objects_trees share the same memory
  Matrix<double> pred_val_mat_train; // n_point * n_tree python self.pred_val_mat_train
  DoubleVector pred_val_sum_train; // python self.pred_val_sum_train
  double lambda_logprior; // python self.lambda_logprior
  
  vector<TreeMCMC_Ptr> p_treemcmcs; //used for cgm

public:

  Bart(const Data& data_train, const Param& param, const Control& control, const Cache& cache, const CacheTemp& cache_temp);
  
  void update_pred_val(UINT i_t, const Data& data_train, const Param& param, const Control& control); // i_t stands for the i-th tree

  void update_pred_val_sum();

  // return {"double mse_tree", "double mse_without_tree", "double var_pred_tree"}
  tuple<double, double, double> compute_mse_within_without(UINT i_t, const Data& data_train);

  void update_residual(UINT tree_id, Data& data_train);

  map<string, DoubleVector_Ptr> predict(const Matrix<double>& x, const DoubleVector& y, const Param& param, const Control& control);

  map<string, DoubleVector_Ptr> predict_train(const Data& data, const Param& param, const Control& control);

  void sample_labels(Data& data, const Param& param); // pg not use

  void sample_lambda_bart(const Data& data, Param& param);

  double compute_train_mse(const Data& data, const Control& control);

  // return {"double loglik_train", "double mse_train"}
  tuple<double, double> compute_train_loglik(const Data& data, const Param& param);

  void countvar(IntVector* counter, Control& control);
};

} // namespace pgbart

namespace pgbart {

double compute_mse(const DoubleVector& y_true, const DoubleVector& y_pred);

void sample_param(Particle_Ptr p_ptr, const Param& param, bool set_to_mean);

void sample_param(TreeMCMC_Ptr treemcmc_ptr, const Param& param, bool set_to_mean);

// convert the mean of labels to zero
void center_labels(Data& data);

void center_labels(Data& data_train, Data& data_test);

void backup_target(Data& data);

} // namespace pgbart

#endif