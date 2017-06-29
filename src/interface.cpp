/****************************************************************
"Bayesian Backfitting" MCMC algorithm for learning additive trees

high level code:
for i in range(n_iterations):
sample lambda_bart_i | T_{*, i-1}, M_{*, i-1} which is just a draw from gamma (or inverse gamma)
for j in trees:
sample T_{j,i} | (T_{j,i-1}, T_{1:j-1, i}, T_{j+1:J, i-1}, M_{1:j-1, i}, M_{j+1:J, i-1})    # M_{j, i} integrated out
- get a single R_j that summarizes the current residual and use this as target
- sample T_{j,i} using PG
- PG: run conditional SMC and draw a single tree sample from the posterior
sample M_{j,i} | (T_{j,i}, T_{1:j-1, i}, T_{j+1:J, i-1}, M_{1:j-1, i}, M_{j+1:J, i-1}) => sample M_{j,i} | T_{j,i} (due to independence structure)
compute predictions of T_{j, i} on both
aggregate preidctions of T_{*, i} for computing final predictions

see paper for further details
http://www.gatsby.ucl.ac.uk/~balaji/pgbart_aistats15.pdf
****************************************************************/

#include <pgbart/include/random.hpp>
#include <pgbart/include/control.hpp>
#include <pgbart/include/data.hpp>
#include <pgbart/include/math.hpp>
#include <pgbart/include/bart.hpp>
#include <pgbart/include/serialize.hpp>
#include <iostream>
#include <ctime>
#include <Rcpp.h>

using namespace std;
using namespace pgbart;
using namespace pgbart::math;
using namespace Rcpp;


pgbart::Matrix<double> convert_matrix(NumericMatrix& rcpp_matrix){
  // convert the R-language matrix to C-language matrix
  UINT n_col = rcpp_matrix.ncol();
  UINT n_row = rcpp_matrix.nrow();
  pgbart::Matrix<double> matrix(n_row, n_col);
  for (UINT i = 0; i < n_row; i++){
    for(UINT j = 0; j < n_col; j++){
      double value = rcpp_matrix(i, j);
      matrix.set(i, j, value);
    }
  }
  return matrix;
}

// the next row is essential to generate the R interface. DON'T EDIT IT
// [[Rcpp::export]]
Rcpp::List train(NumericMatrix& train_data, NumericVector& train_label, bool if_test, NumericMatrix& test_data, NumericVector& test_label, String& model_file, double alpha_bart, double alpha_split,
  double beta_split, bool if_center_label, bool if_debug, double ess_threshold, unsigned int init_seed_id, bool if_set_seed, double k_bart,
  unsigned int m_bart, unsigned int min_size, unsigned int ndpost, unsigned int nskip, unsigned int keepevery, const String& variance_type, double q_bart,
  unsigned int verbose_level, unsigned int n_particles, const String& resample_type) {
	/*
		Train the model and save it in a file using the given training data and parameters

		Parmeters
		---------
		train_data : the set of features of the training data
		train_label : the set of the label of the training data
		model_file : decide where to save the model
		alpha_bart : the df parameter in BART
		alpha_split : the parameter that influences the split prob of a tree. Reference to expression (4) in section 2.4 of the paper
		beta_split : the parameter that influences the split prob of a tree. Reference to expression (4) in section 2.4 of the paper
		if_center_label : whether to resize the label between [-1, +1]
		if_debug : print more debug info
		ess_threshold : threshold for resample
		init_seed_id : set the random seed
		if_set_seed : whether to make the random seed effect
		k_bart : hyperparameter in section 2.4 of the paper
		m_bart : number of trees in the model
		min_size : a node that can split must have at least (min_size + 1) data points
		ndpost :  the number of posterior draws after burn in
		nskip :  Number of MCMC iterations to be treated as burn in
		keepevery : Every keepevery draw is kept to be returned to the user
		variance_type : variance use for setting hyperparameters
		q_bart : controls the prior over sigma^2 in BART
		verbose_level : level of detail about the intermediate process
		n_particles : number of particles to use in the PG_Sampler. Reference to section 3.3 of the paper
	*/

  // collect all the parameters in an object
  Control control(alpha_bart, alpha_split, beta_split, if_center_label, if_debug, ess_threshold, init_seed_id, if_set_seed,
    k_bart, m_bart, min_size, ndpost, nskip, keepevery, variance_type, q_bart, verbose_level, n_particles, resample_type);

  cout << control.toString() << endl;
  if (if_set_seed){
    cout << "Setting random seed_id = " << init_seed_id << endl;
    set_random_seed(init_seed_id);
  }
  else{
    cout << "Not Setting random seed" << endl;
  }

  cout << "Loading data..." << endl;
  pgbart::Matrix<double> x = convert_matrix(train_data);
  std::vector<double> y_original = as<std::vector<double>>(train_label);
  Data data_train(x, y_original);
  cout << "data_train point: " << data_train.n_point << endl;
  cout << "data_train features: " <<data_train.n_feature << endl;
  Data data_test;
  if (if_test){
	  pgbart::Matrix<double> test_x = convert_matrix(test_data);
	  std::vector<double> test_y_original = as<std::vector<double>>(test_label);
	  Data tmp_test(test_x, test_y_original);
	  data_test = tmp_test;
	  cout << "data_test point: " << data_test.n_point << endl;
	  cout << "data_test features: " << data_test.n_feature << endl;
  }
  cout << "Loading data...completed!" << endl;

  double train_mean = 0;
  if (control.if_center_label) {
    cout << "Centering the labels!" << endl;
	// center the label between
  	if (if_test){
  		center_labels(data_train, data_test);
  	} else {
  		train_mean = math::mean(data_train.y_original);
  		center_labels(data_train);
  	}
  }

  // create an extra copy of the label
  backup_target(data_train);

  Model pgbart_model(train_mean, control.m_bart, control.ndpost / control.keepevery);

  UINT train_length = control.ndpost / control.keepevery;
  if (control.ndpost % control.keepevery != 0) {
    train_length = control.ndpost / control.keepevery + 1;
  }

  UINT test_length = train_length;
  if (!if_test) {
    test_length = 0;
  }
  // initialize the return object
  Rcpp::List status_result(0);
  Rcpp::NumericMatrix matrix_yhat_train(train_length, data_train.x.n_row); UINT i_yhat_train = 0;
  Rcpp::NumericMatrix matrix_yhat_test(test_length, data_test.x.n_row); UINT i_yhat_test = 0;
  Rcpp::NumericVector vector_mse_train(train_length); UINT i_mse_train = 0;
  Rcpp::NumericVector vector_mse_test(test_length); UINT i_mse_test = 0;
  Rcpp::NumericVector vector_loglik_train(train_length); UINT i_loglik_train = 0;
  Rcpp::NumericVector vector_loglik_test(test_length); UINT i_loglik_test = 0;
  Rcpp::NumericVector vector_sigma(train_length); UINT i_sigma = 0;
  Rcpp::NumericVector vector_first_sigma(train_length); UINT i_first_sigma = 0;
  Rcpp::NumericMatrix matrix_varcount(train_length, data_train.x.n_column); UINT i_varcount = 0;

  // find all the potential split point
  // and initialize the parameters of the training process
  Param_Ptr param_ptr; Cache_Ptr cache_ptr; CacheTemp_Ptr cache_temp_ptr;
  tie(param_ptr, cache_ptr, cache_temp_ptr) = precompute(data_train, control);
  Param& param = *param_ptr; Cache& cache = *cache_ptr; CacheTemp& cache_temp = *cache_temp_ptr;

  // initialize the bart model
  Bart bart(data_train, param, control, cache, cache_temp);

  bool change = true;

  IntVector tree_order(control.m_bart);
  for (UINT i = 0; i < control.m_bart; i++) {
    tree_order[i] = i;
  }

  cout << "initial settings: " << endl;
  cout << "lamda_bart = " << param.lambda_bart << endl;

  // conpute mse and loglik for the initial status
  double loglik_train, mse_train;
  tie(loglik_train, mse_train) = bart.compute_train_loglik(data_train, param);
  cout << "mse_train = " << mse_train << ", loglik_train = " << loglik_train << endl;
  UINT total_iteration = control.ndpost + control.nskip;
  // starting iterations
  cout << "burning in..." << endl;
  for (UINT itr = 0; itr < total_iteration; itr++) {
    double logprior = 0.0;
    // sampling lambda_bart
    bart.sample_lambda_bart(data_train, param);

  	if (itr < nskip && itr % control.keepevery == 0) {
  		vector_first_sigma[i_first_sigma] = param.lambda_bart;
  		i_first_sigma++;
  	}

  	if (itr >= nskip && (itr - nskip) % control.keepevery == 0) {
  		vector_sigma[i_sigma] = param.lambda_bart;
  		i_sigma++;
  	}

    logprior += bart.lambda_logprior;

	   // sample each tree by random order
    tree_order = shuffle(const_cast<IntVector&>(tree_order));

    for (auto tree_id : tree_order) {

      if (control.if_debug) {
        cout << "\ntree_id = " << tree_id << endl;
      }
      // update residual
      if (control.verbose_level >= 1)
        cout << "\n\ncompute residuals for the "<< tree_id << "-th tree!  " << itr << endl;

	  // update residuals for i_t-th tree

      bart.update_residual(tree_id, data_train);
      update_cache_temp(cache_temp, cache, data_train, param, control);
      // MCMC for i_t-th tree
      bart.p_particles[tree_id]->update_loglik_node_all(data_train, param, cache, control);
      tie(bart.p_particles[tree_id], change) = run_mcmc_single_tree(bart.p_particles[tree_id], control, data_train, param,
          cache, change, cache_temp, bart.pmcmc_objects[tree_id]);
	  // sample mu for i_t-th tree
      sample_param(bart.p_particles[tree_id], param, false);
      logprior += bart.p_particles[tree_id]->pred_val_logprior;

      //update pred_val
      bart.update_pred_val(tree_id, data_train, param, control);

      //update stats
      //'change' indicates whether MCMC move was accepted
      bart.p_particles[tree_id]->update_depth();

    }

    logprior = -__DBL_MAX__;

  	if (itr >= nskip && (itr - nskip) % keepevery == 0) {
  		time_t t = time(NULL);
  		struct tm* current_time = localtime(&t);
  		printf("[%d/%d/%d %d %d %d]",
  			1 + current_time->tm_mon,
  			current_time->tm_mday,
  			1900 + current_time->tm_year,
  			current_time->tm_hour,
  			current_time->tm_min,
  			current_time->tm_sec);
  		std::tie(loglik_train, mse_train) = bart.compute_train_loglik(data_train, param);
  		vector_mse_train[i_mse_train] = mse_train; i_mse_train++;
  		vector_loglik_train[i_loglik_train] = loglik_train; i_loglik_train++;
  		cout << "itr: " << itr - nskip + 1 << "  mse_train = " << mse_train << ", loglik_train = " << loglik_train << endl;

		pgbart_model.add_itr(bart);

      for (UINT j = 0; j < data_train.x.n_row; j++) {
        matrix_yhat_train(i_yhat_train, j) = bart.pred_val_sum_train[j];
      }

      i_yhat_train++;

      IntVector* counter = new IntVector(data_train.x.n_column, 0);
      bart.countvar(counter, control);
      for (UINT j = 0; j < data_train.x.n_column; j++) {
        matrix_varcount(i_varcount, j) = counter->at(j);
      }
      i_varcount++;
      delete counter;
  	}

  	if (if_test && itr >= nskip && (itr - nskip) % keepevery == 0) {
  		cout << "\n" << (string("*") * 30U) << " Test for BART iteration = " << itr - control.nskip + 1 << (string("*") * 30U) << endl;

  		//save the return list
  		UINT n_point = data_test.x.n_row;
  		// prepare for the predicting
  		pgbart::DoubleVector pred_result(n_point, 0);
  		// for each tree of the model, compute the pred_value
  		for (UINT m = 0; m < bart.p_particles.size(); m++) {
  			IntVector* leaf_id = bart.p_particles[m]->gen_rules_tree(data_test);
  			pgbart::DoubleVector* temp = bart.p_particles[m]->predict_real_val_fast(leaf_id);
  			for (UINT k = 0; k < n_point; k++) {
  				pred_result[k] += temp->at(k);
        }
  			delete leaf_id;
  			delete temp;
  		}
  		double mse_test = sum2(pred_result - data_test.y_original) / n_point;
  		double loglik_test = 0.5 * n_point * (std::log(param.lambda_bart) - std::log(2 * PI) - param.lambda_bart * mse_test);
  		vector_mse_test[i_mse_test] = mse_test; i_mse_test++;
  		vector_loglik_test[i_loglik_test] = loglik_test; i_loglik_test++;

  		cout << "mse_test = " << mse_test << "   ";
  		cout << "loglik_test = " << loglik_test << endl;

      for (UINT j = 0; j < data_test.x.n_row; j++) {
        matrix_yhat_test(i_yhat_test, j) = pred_result[j];
      }
      i_yhat_test++;
  	}
  }

  status_result["yhat.train"] = matrix_yhat_train;
  status_result["yhat.test"] = matrix_yhat_test;
  status_result["mse.train"] = vector_mse_train;
  status_result["loglik.train"] = vector_loglik_train;
  status_result["mse.test"] = vector_mse_test;
  status_result["loglik.test"] = vector_loglik_test;
  status_result["first.sigma"] = vector_first_sigma;
  status_result["sigma"] = vector_sigma;
  status_result["varcount"] = matrix_varcount;

  //save model to a file

  save_model(model_file, pgbart_model);

  return status_result;
}


// the next row is essential to generate the R interface. DON'T EDIT IT
// [[Rcpp::export]]
NumericVector predict(NumericMatrix& test_data, String& model_file){
  /*
	Predict the output label using the existed model

	Parmeter
	--------
	predict_data : the input features
	model_file : the path where the model is saved
  */
  Model model = load_model(model_file);
  UINT n_point = test_data.nrow();
  UINT step_length = test_data.ncol();

  // prepare for the predicting
  pgbart::DoubleVector pred_result(n_point, 0);
  pgbart::Matrix<double> matrix = convert_matrix(test_data);
  pgbart::DoubleVector& x_test = matrix.elements;

  Rcpp::NumericMatrix matrix_yhat_predict(model.n_iteration, n_point);

  // for each row of the input data, predict the output
  for (UINT k = 0; k < n_point; k++){
	// for each tree of the model, compute the pred_value
	  for (UINT itr = 0; itr < model.n_iteration; itr++){
	    pred_result[k] = 0;
		  for (UINT m = 0; m < model.m_bart; m++){
			  BartTree& tree = model.trees[itr][m];
			  UINT node_id = 0;
			  // traverse to a leaf node
			  while (true){
				  if (check_if_included(tree.leaf_node_ids, node_id)){
					  break;
				  }
				  UINT left = node_id * 2 + 1;
				  UINT right = node_id * 2 + 2;
				  if (x_test[k * step_length + tree.node_info[node_id].feat_id_chosen] <= tree.node_info[node_id].split_chosen)
					  node_id = left;
				  else
					  node_id = right;
			  }
			  pred_result[k] += tree.pred_val[node_id];
		  }
		  pred_result[k] += model.train_mean;
		  matrix_yhat_predict(itr, k) = pred_result[k];
	  }
  }
  return matrix_yhat_predict;
}
