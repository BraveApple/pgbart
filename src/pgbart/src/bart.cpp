#include <iostream>

#include "pgbart/include/bart.hpp"

using namespace pgbart::math;

namespace pgbart {

Bart::Bart(const Data& data_train, const Param& param, const Control& control, const Cache& cache, const CacheTemp& cache_temp)
  : pred_val_mat_train(data_train.n_point, control.m_bart) {
  this->p_particles.clear();
  this->pmcmc_objects.clear();
  this->update_pred_val_sum();
  for (UINT i = 0; i < control.m_bart; i++) {
	  if (control.mcmc_type == "pg") {
		  Pmcmc_Ptr pmcmc_ptr = init_particle_mcmc(data_train, control, param, cache, cache_temp);
		  Particle_Ptr p_ptr = pmcmc_ptr->p_ptr;
		  sample_param(p_ptr, param, false);
		  this->p_particles.push_back(p_ptr);
		  this->pmcmc_objects.push_back(pmcmc_ptr);
	}
	  else if (control.mcmc_type == "cgm") {
		  TreeMCMC_Ptr treemcmc_ptr = init_cgm_mcmc(data_train, control, param, cache, cache_temp);
		  sample_param(treemcmc_ptr, param, false);
		  this->p_treemcmcs.push_back(treemcmc_ptr);
	}
    // set the initial pred value
    this->update_pred_val(i, data_train, param, control);
  }
  // compute lambda_logprior
  this->lambda_logprior = compute_gamma_loglik(param.lambda_bart, param.alpha_bart, param.beta_bart);
}

void Bart::update_pred_val(UINT i_t, const Data& data_train, const Param& param, const Control& control) {
	// IntVector* leaf_id;
	// DoubleVector* temp;
  IntVector_Ptr leaf_id_ptr = nullptr;
  DoubleVector_Ptr temp_ptr = nullptr;
	if (control.mcmc_type == "pg") {
		leaf_id_ptr = this->p_particles[i_t]->gen_rules_tree(data_train);
		temp_ptr = this->p_particles[i_t]->predict_real_val_fast(leaf_id_ptr);
	}
	else if (control.mcmc_type == "cgm") {
		leaf_id_ptr = this->p_treemcmcs[i_t]->gen_rules_tree(data_train);
		temp_ptr = this->p_treemcmcs[i_t]->predict_real_val_fast(leaf_id_ptr);
	}
	if (control.if_debug) {
		std::cout << "current data_train.y_residual = " << toString(data_train.y_residual) << std::endl;
		std::cout << "predictions of current tree = " << toString(*temp_ptr) << std::endl;
	}
	this->pred_val_sum_train += *temp_ptr;
	this->pred_val_sum_train -= pred_val_mat_train(":", i_t);
    this->pred_val_mat_train.set(":", i_t, *temp_ptr);
	// delete leaf_id;
	// delete temp;

	if (control.if_debug) {
		std::cout << "current data_train.y_original = " << toString(data_train.y_original) << std::endl;
		std::cout << "predictions of current bart = " << toString(pred_val_sum_train) << std::endl;
	}
}

void Bart::update_pred_val_sum() {
  pred_val_sum_train = pred_val_mat_train.sum(0);
} 

/*
return {"double mse_tree", "double mse_without_tree", "double var_pred_tree"}
*/
tuple<double, double, double> Bart::compute_mse_within_without(UINT i_t, const Data& data_train) {
  DoubleVector pred_tree = pred_val_mat_train(i_t, ":");
  double var_pred_tree = variance(pred_tree);
  DoubleVector pred_without_tree = pred_val_sum_train - pred_val_mat_train(i_t, ":");
  double mse_tree = compute_mse(data_train.y_original, pred_tree);
  double mse_without_tree = compute_mse(data_train.y_original, pred_without_tree);
  return make_tuple(mse_tree, mse_without_tree, var_pred_tree);
}

void Bart::update_residual(UINT tree_id , Data& data_train) {
  data_train.y_residual = data_train.y_original - pred_val_sum_train + pred_val_mat_train(":", tree_id);
}

map<string, DoubleVector_Ptr> Bart::predict(const Matrix<double>& x, const DoubleVector& y, const Param& param, const Control& control) {
  DoubleVector_Ptr pred_prob_ptr(new DoubleVector(x.n_row, 0.0));
  DoubleVector& pred_prob = *pred_prob_ptr;
  DoubleVector_Ptr pred_val_ptr(new DoubleVector(x.n_row, 0.0));
  DoubleVector& pred_val = *pred_val_ptr;
  double log_const = 0.5 * std::log(param.lambda_bart) - 0.5 * std::log(2 * BART_PI);
  for (UINT tree_id = 0; tree_id < control.m_bart; tree_id++) {
	  if (control.mcmc_type == "pg") {
		  IntVector leaf_node_ids = this->p_particles[tree_id]->tree_ptr->traverse(x); // apply rules to "x" and create "leaf_node_ids"
		  pred_val += at(this->p_particles[tree_id]->tree_ptr->pred_val_n, leaf_node_ids);
	  }
	  else if (control.mcmc_type == "cgm") {
		  IntVector leaf_node_ids = this->p_treemcmcs[tree_id]->tree_ptr->traverse(x); // apply rules to "x" and create "leaf_node_ids"
		  pred_val += at(this->p_treemcmcs[tree_id]->tree_ptr->pred_val_n, leaf_node_ids);
	  }
  }
  pred_prob = exp(-0.5 * param.lambda_bart * square(y - pred_val) + log_const);
  map<string, DoubleVector_Ptr> map_temp;
  map_temp["pred_mean"] = pred_val_ptr;
  map_temp["pred_prob"] = pred_prob_ptr;
  return map_temp;
}

map<string, DoubleVector_Ptr> Bart::predict_train(const Data& data, const Param& param, const Control& control)
{
  DoubleVector& pred_val = this->pred_val_sum_train;
  double log_const = 0.5 * std::log(param.lambda_bart) - 0.5 * std::log(2 * BART_PI);
  DoubleVector loglik = -0.5 * param.lambda_bart * square(data.y_original - pred_val) + log_const;
  DoubleVector pred_prob = exp(loglik);
  map<string, DoubleVector_Ptr> map_temp;
  map_temp["pred_mean"] = make_shared<DoubleVector>(pred_val);
  map_temp["pred_prob"] = make_shared<DoubleVector>(pred_prob);
  return map_temp;
}

void Bart::sample_labels(Data& data, const Param& param) {
  data.y_original = pred_val_sum_train + simulate_normal_distribution(0.0, 1.0, data.n_point) / std::sqrt(param.lambda_bart);
}

void Bart::sample_lambda_bart(const Data& data, Param& param) {
  double lambda_alpha = param.alpha_bart + 0.5 * data.n_point;
  double lambda_beta = param.beta_bart + 0.5 * sum2(data.y_original - pred_val_sum_train);
  param.lambda_bart = simulate_gamma_distribution(lambda_alpha, 1.0 / lambda_beta);
  param.log_lambda_bart = std::log(param.lambda_bart);
  this->lambda_logprior = compute_gamma_loglik(param.lambda_bart, param.alpha_bart, param.beta_bart);
}

double Bart::compute_train_mse(const Data& data, const Control& control) {
  // NOTE: inappropriate if y_train_orig contains noise
  double mse_train = compute_mse(data.y_original, this->pred_val_sum_train);
  if (control.verbose_level >= 1)
    std::cout << "mse_train = " << mse_train << std::endl;
  return mse_train;
}

// return {"double loglik_train", "double mse_train"}
tuple<double, double> Bart::compute_train_loglik(const Data& data, const Param& param) {
  double mse_train = compute_mse(data.y_original, this->pred_val_sum_train);
  double loglik_train = 0.5 * data.n_point * (std::log(param.lambda_bart)
    - std::log(2 * BART_PI) - param.lambda_bart * mse_train);
  return make_tuple(loglik_train, mse_train);
}

//count the var used times
void Bart::countvar(IntVector* vector, Control& control){
  for (UINT m = 0; m < control.m_bart; m++){
	  if (control.mcmc_type == "pg") {
		  map<UINT, SplitInfo>& temp_node_info = this->p_particles[m]->node_info;
		  for (auto pair : temp_node_info) {
			  SplitInfo& temp = pair.second;
			  vector->at(temp.feat_id_chosen) = vector->at(temp.feat_id_chosen) + 1;
		  }
	  }
	  else if (control.mcmc_type == "cgm") {
		  map<UINT, SplitInfo>& temp_node_info = this->p_treemcmcs[m]->node_info;
		  for (auto pair : temp_node_info) {
			  SplitInfo& temp = pair.second;
			  vector->at(temp.feat_id_chosen) = vector->at(temp.feat_id_chosen) + 1;
		  }
	  }
  }
}

} // nmaespace pgbart

namespace pgbart {

double compute_mse(const DoubleVector& y_true, const DoubleVector& y_pred) {
  return sum2(y_true - y_pred) / y_true.size();
}

void sample_param(Particle_Ptr p_ptr, const Param& param, bool set_to_mean) {
  /*
  prediction at node (draw from posterior over leaf parameter);
  Note that prediction uses posterior mean
  */
  math::initial_map(p_ptr->tree_ptr->pred_val_n, p_ptr->tree_ptr->leaf_node_ids, BART_DBL_MAX);
  p_ptr->pred_val_logprior = 0.0;
  for (auto node_id : p_ptr->tree_ptr->leaf_node_ids) {
    double mu_mean_post = p_ptr->mu_mean_post[node_id];
    double mu_prec_post = p_ptr->mu_prec_post[node_id];
    if (set_to_mean)
      p_ptr->tree_ptr->pred_val_n[node_id] = 0.0 + mu_mean_post;
    else {
      p_ptr->tree_ptr->pred_val_n[node_id] = simulate_normal_distribution(0.0, 1.0) / std::sqrt(mu_prec_post) + mu_mean_post;
    }
    p_ptr->pred_val_logprior += compute_normal_loglik(p_ptr->tree_ptr->pred_val_n[node_id], param.mu_mean, param.mu_prec);
  }
}

void sample_param(TreeMCMC_Ptr t_ptr, const Param& param, bool set_to_mean) {
	math::initial_map(t_ptr->tree_ptr->pred_val_n, t_ptr->tree_ptr->leaf_node_ids, BART_DBL_MAX);
	t_ptr->pred_val_logprior = 0.0;
	for (auto node_id : t_ptr->tree_ptr->leaf_node_ids) {
		double mu_mean_post = t_ptr->mu_mean_post[node_id];
		double mu_prec_post = t_ptr->mu_prec_post[node_id];
		if (set_to_mean)
			t_ptr->tree_ptr->pred_val_n[node_id] = 0.0 + mu_mean_post;
		else {
			t_ptr->tree_ptr->pred_val_n[node_id] = simulate_normal_distribution(0.0, 1.0) / std::sqrt(mu_prec_post) + mu_mean_post;
		}
		t_ptr->pred_val_logprior += compute_normal_loglik(t_ptr->tree_ptr->pred_val_n[node_id], param.mu_mean, param.mu_prec);
	}
}

// convert the mean of labels to zero
void center_labels(Data& data) {
  double y_train_mean = mean(data.y_original);
  data.y_original -= y_train_mean;
}

void center_labels(Data& data_train, Data& data_test){
  double y_train_mean = mean(data_train.y_original);
  data_train.y_original -= y_train_mean;
  data_test.y_original -= y_train_mean;
}

void backup_target(Data& data) {
  data.y_residual = data.y_original;
}

} // namespace pgbart
