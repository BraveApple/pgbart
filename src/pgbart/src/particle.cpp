#include <exception>
#include <iostream>

#include "pgbart/include/particle.hpp"

using namespace pgbart::math;

namespace pgbart {

//Particle::Particle(Tree* tree, IntVector& train_ids, Param& param, CacheTemp& cache_temp)
Particle::Particle(const IntVector& train_ids, const Param& param, const CacheTemp& cache_temp)
  :State(train_ids, param, cache_temp) {
  this->ancestry.clear();
  this->nodes_processed_itr.clear();
  this->grow_nodes_itr.clear();
  this->log_sis_ratio_d.clear();
  if (cache_temp.isexisted){
    this->do_not_grow = false;
    this->grow_nodes.push_back(0);
  }
  this->isexisted = true;
}

Particle::Particle(const Particle& p) {
  this->tree_ptr = Tree_Ptr(new Tree(p.tree_ptr));
  this->tree_ptr->leaf_node_ids = p.tree_ptr->leaf_node_ids;
  this->do_not_split = p.do_not_split; // insert a pair by key
  this->sum_y = p.sum_y;
  this->sum_y2 = p.sum_y2;
  this->n_points = p.n_points;
  this->loglik = p.loglik;
  this->mu_mean_post = p.mu_mean_post;
  this->mu_prec_post = p.mu_prec_post;
  this->train_ids = p.train_ids;
  this->node_info = p.node_info;
  this->logprior = p.logprior;
  this->loglik_current = p.loglik_current;
  this->ancestry = p.ancestry;
  this->nodes_processed_itr = p.nodes_processed_itr;
  this->grow_nodes_itr = p.grow_nodes_itr;
  this->log_sis_ratio_d = p.log_sis_ratio_d;
  this->do_not_grow = p.do_not_grow;
  this->grow_nodes = p.grow_nodes;
  this->isexisted = p.isexisted;
}

double Particle::process_node_id(const Data& data_train, const Param& param, const Control& control, const Cache& cache, const UINT& node_id, Random& pgrandom) {
  double log_sis_ratio;
  if (do_not_split[node_id])
    log_sis_ratio = 0.0;
  else {
    IntVector& train_ids = this->train_ids[node_id];
    UINT left_node_id = this->tree_ptr->getLeftNodeID(node_id);
    UINT right_node_id = this->tree_ptr->getRightNodeID(node_id);

    if (control.verbose_level >= 4)
      std::cout << "train_ids for this node = " << math::toString(train_ids) << std::endl;

    bool do_not_split_node_id;
    SplitInfo_Ptr split_info_ptr;
    double logprior_nodeid;
    IntVector_Ptr train_ids_left_ptr;
    IntVector_Ptr train_ids_right_ptr;
    CacheTemp_Ptr cache_temp_ptr;

    // find the split feature and split value equiprobably
    tie(do_not_split_node_id, split_info_ptr, log_sis_ratio, logprior_nodeid, train_ids_left_ptr, train_ids_right_ptr, cache_temp_ptr) =
      this->prior_proposal(data_train, node_id, train_ids, param, cache, control, pgrandom);

    if (do_not_split_node_id)
      do_not_split[node_id] = true;
    else {
      // update the statistics for the two new node
      update_left_right_statistics(node_id, logprior_nodeid, *split_info_ptr, *cache_temp_ptr,
        control, *train_ids_left_ptr, *train_ids_right_ptr, data_train, cache, param);
      this->grow_nodes.push_back(left_node_id);
      this->grow_nodes.push_back(right_node_id);
    }
  }
  return log_sis_ratio;
}

void Particle::grow_next(const Data& data_train, const Param& param, const Control& control, const Cache& cache, Random& pgrandom)
{
  /*
  grows just one node at a time (nodewise expansion)
  breaks after processing the first non do_not_grow node or when grow_nodes is empty
  Note that multiple nodes could be killed in a single grow_next call
  */

  bool do_not_grow_temp = true;
  double log_sis_ratio_temp = 0.0;
  IntVector node_ids_processed;

  if (this->grow_nodes.empty()) {
    if (control.verbose_level >= 2) {
      std::cout << "None of the leaves can be grown any further: Current depth = " 
        << this->tree_ptr->getTreeDepth() << ", Skipping grow_next!" << std::endl; 
    }
      
  }
  else {
    while (true) {
      /*
      loop through current leaf nodes, process first "non do_not_grow" node and break;
      if none of the nodes can be processed, do_not_grow = True
      */

      UINT remove_position = 0; // just pop the oldest node
      UINT node_id = this->grow_nodes[remove_position];
      delete_id(this->grow_nodes, remove_position);
      node_ids_processed.push_back(node_id);
      do_not_grow_temp = do_not_grow_temp && do_not_split[node_id];
      if (do_not_split[node_id]) {
        if (control.verbose_level >= 3)
          std::cout << "Skipping split at node_id = " << node_id << std::endl;
        if (this->grow_nodes.empty())
          break;
      }
      else {
        // do the split action
        log_sis_ratio_temp += process_node_id(data_train, param, control, cache, node_id, pgrandom);
        break; // you have processed a non do_not_grow node, take a break!
      }
    }
    loglik_current = compute_loglik();
  }

  this->log_sis_ratio = log_sis_ratio_temp;
  this->do_not_grow = do_not_grow_temp;
  if (!node_ids_processed.empty())
    nodes_processed_itr.push_back(node_ids_processed);
}

void Particle::check_nodes_processed_itr() {
  set<UINT> set_temp;
  for (auto nodes : nodes_processed_itr) {
    for (auto node_id : nodes) {
      if (check_if_included(set_temp, node_id))
        std::cout << "node_id = " << node_id << "present multiple times in nodes_processed_itr!" << std::endl;
      else
        set_temp.insert(node_id);
    }
  }
}



} // namespace pgbart

namespace pgbart {

/*
return {"double log_pd", "double ess", "DoubleVector log_weights_new", "DoubleVector weights_norm_new"}
*/
tuple<double, double, DoubleVector_Ptr>
  update_particle_weights(Vec_Particle_Ptr particles_ptr, DoubleVector_Ptr log_weights_ptr, const Control& control) {
  DoubleVector_Ptr weights_norm_new_ptr(new DoubleVector());
  UINT length = particles_ptr->size();
  for (UINT p_id = 0; p_id < length; p_id++) {
    Particle_Ptr p_ptr = (*particles_ptr)[p_id];
    if (control.verbose_level >= 2) {
      std::cout << "p_id = " << p_id << ", log_sis_ratio =  " << p_ptr->log_sis_ratio << std::endl;
    }

    (*log_weights_ptr)[p_id] += p_ptr->log_sis_ratio;
  }

  *weights_norm_new_ptr = softmax(*log_weights_ptr); // normalized weights
  double ess = 1.0 / sum2(*weights_norm_new_ptr) / control.n_particles;
  double log_pd = log_sum_exp(*log_weights_ptr);
  return make_tuple(log_pd, ess, weights_norm_new_ptr);
}

Vec_Particle_Ptr resample(const Vec_Particle_Ptr particles_ptr, DoubleVector_Ptr log_weights_ptr, const Control& control,
  const double& log_pd, const double& ess, DoubleVector_Ptr weights_norm_ptr, Particle_Ptr tree_pg_ptr, Random& pgrandom) {
  
  IntVector_Ptr pid_list_ptr;
  if (ess <= control.ess_threshold) {
    if (tree_pg_ptr->isexisted) {
      // resample particles by the probability computed before
      // note that the first particle is the old tree except there is no old tree
      pid_list_ptr = IntVector_Ptr(new IntVector());
      pid_list_ptr = resample_pids_basic(control, control.n_particles - 1, weights_norm_ptr, pgrandom);
      pgrandom.shuffle(*pid_list_ptr); // shuffle so that particle is assigned randomly
      pid_list_ptr->insert(pid_list_ptr->begin(), 0);
    }
    else {
      pid_list_ptr = resample_pids_basic(control, control.n_particles, weights_norm_ptr, pgrandom);
    }
    *log_weights_ptr = ones<double>(control.n_particles) * (log_pd - std::log(control.n_particles));
  }
  else {
    pid_list_ptr = make_shared<IntVector>(range(0U, control.n_particles));
  }
  if (control.verbose_level >= 2) {
    std::cout << "ess = " << ess << ", ess_threshold = " << control.ess_threshold << std::endl;
    std::cout << "new particle ids = \n" << toString(*pid_list_ptr) << std::endl;
  }

  // copy the particles to new memory
  Vec_Particle_Ptr op_ptr = create_new_particles(particles_ptr, pid_list_ptr, control);
  
  UINT length = pid_list_ptr->size();
  for (UINT i = 0; i < length; i++) {
    (*op_ptr)[i]->ancestry.push_back((*pid_list_ptr)[i]);
  }
  return op_ptr;
}

IntVector_Ptr resample_pids_basic(const Control& control, const UINT& n_particles, DoubleVector_Ptr probs_ptr, Random& pgrandom) {
  IntVector_Ptr pid_list_ptr;
  if (control.resample_type == "multinomial")
    pid_list_ptr = sample_multinomial_particle(n_particles, probs_ptr, pgrandom);
  else if (control.resample_type == "systematic")
    pid_list_ptr = sample_systematic_particle(n_particles, probs_ptr, pgrandom);
  else {
    std::cout << "control.resample_type must be \"multinomial\" or \"systematic\"!" << std::endl;
    exit(1);
  }
  return pid_list_ptr;
}

IntVector_Ptr sample_multinomial_particle(const UINT& n_particles, DoubleVector_Ptr probs_ptr, Random& pgrandom) {
  IntVector vec_time = pgrandom.sample_multinomial_distribution(*probs_ptr, n_particles);
  IntVector_Ptr pid_list_ptr(new IntVector());
  for (size_t i = 0; i < vec_time.size(); i++) {
    UINT n_time = vec_time[i];
    for (size_t j = 0; j < n_time; j++)
      pid_list_ptr->push_back(i);
  }
  return pid_list_ptr;
}

IntVector_Ptr sample_systematic_particle(const UINT& n_particles, DoubleVector_Ptr probs_ptr, Random& pgrandom) {
  /*
  systematic re-sampling algorithm.
  Note: objects with > 1/n probability (better than average) are guaranteed to occur atleast once.
  see section 2.4 of 'Comparison of Resampling Schemes for Particle Filtering' by Douc et. al for more info.
  */
  if (n_particles != probs_ptr->size()) {
    std::cout << n_particles << "!=" << probs_ptr->size() << std::endl;
    exit(1);
  }

  if (absolute(sum(*probs_ptr) - 1) < 1e-10) {
    std::cout << "the sum of probs must be equal to one!" << std::endl;
    exit(1);
  }

  DoubleVector cum_probs = cumsum(*probs_ptr);
  double u = pgrandom.simulate_continuous_uniform_distribution(0.0, 1.0) / n_particles;
  UINT i = 0;
  IntVector_Ptr indices_ptr(new IntVector());
  while (true) {
    while (u > cum_probs[i])
      i++;
    indices_ptr->push_back(i);
    u += 1.0 / n_particles;
    if (u > 1)
      break;
  }
  return indices_ptr;
}

Vec_Particle_Ptr create_new_particles(const Vec_Particle_Ptr particles_ptr, IntVector_Ptr pid_list_ptr, const Control& control) {
  set<UINT> list_allocated;
  UINT length = pid_list_ptr->size();
  UINT temp_pid;
  Vec_Particle_Ptr op_ptr(new vector<Particle_Ptr>());
  for (UINT i = 0; i < length; i++) {
    temp_pid = pid_list_ptr->at(i);
    if (list_allocated.find(temp_pid) != list_allocated.end()) {
      op_ptr->push_back(copy_particle((*particles_ptr)[temp_pid]));
    }
    else {
      op_ptr->push_back((*particles_ptr)[temp_pid]);
    }
    list_allocated.insert(temp_pid);
  }
  return op_ptr;
}

Particle_Ptr copy_particle(Particle_Ptr particle_ptr) {
  Particle_Ptr op(new Particle());
  //lists
  op->tree_ptr->leaf_node_ids = particle_ptr->tree_ptr->leaf_node_ids;
  op->tree_ptr->non_leaf_node_ids = particle_ptr->tree_ptr->non_leaf_node_ids;
  op->ancestry = particle_ptr->ancestry;
  op->nodes_processed_itr = particle_ptr->nodes_processed_itr;
  op->grow_nodes = particle_ptr->grow_nodes;
  op->grow_nodes_itr = particle_ptr->grow_nodes_itr;
  //dicts
  op->do_not_split = particle_ptr->do_not_split;
  op->log_sis_ratio_d = particle_ptr->log_sis_ratio_d;
  op->sum_y = particle_ptr->sum_y;
  op->sum_y2 = particle_ptr->sum_y2;
  op->n_points = particle_ptr->n_points;

  op->mu_mean_post = particle_ptr->mu_mean_post;
  op->mu_prec_post = particle_ptr->mu_prec_post;

  op->train_ids = particle_ptr->train_ids;
  op->node_info = particle_ptr->node_info;
  op->loglik = particle_ptr->loglik;
  op->logprior = particle_ptr->logprior;
  // other variables
  op->tree_ptr->tree_depth = particle_ptr->tree_ptr->tree_depth;
  op->do_not_grow = particle_ptr->do_not_grow;
  op->loglik_current = particle_ptr->loglik_current;

  return op;
}

tuple<Vec_Particle_Ptr, DoubleVector_Ptr> init_particles(const Data& data_train, const Control& control, const Param& param, const CacheTemp cache_temp) {
  Vec_Particle_Ptr particles_ptr(new vector<Particle_Ptr>());
  DoubleVector_Ptr weights_ptr(new DoubleVector());

  UINT length = control.n_particles;
  UINT n_train = data_train.n_point;
  IntVector train_ids(n_train, 0);
  for (UINT j = 1; j < n_train; j++) {
    train_ids[j] = j;
  }

  double log_n_particles = std::log(control.n_particles);
  for (UINT i = 0; i < length; i++) {
    Particle_Ptr particle_ptr(new Particle(train_ids, param, cache_temp));
    particles_ptr->push_back(particle_ptr);
    weights_ptr->push_back(particle_ptr->loglik[0] - log_n_particles);
  }
  return make_tuple(particles_ptr, weights_ptr);
}

tuple<Vec_Particle_Ptr, double, DoubleVector_Ptr, double>
  run_smc(Vec_Particle_Ptr particles_ptr, const Data& data, const Control& control, const Param& param,
  DoubleVector_Ptr log_weights_ptr, const Cache& cache, Particle_Ptr tree_pg_ptr, Random& pgrandom) {
  // See more information about this function in Algorithm 2 of the PGBART paper
  
  double log_pd = 0.0;
  double ess = 0.0;
  DoubleVector_Ptr weights_norm_ptr;

  if (control.verbose_level >= 2) {
    std::cout << "Conditioned tree: " << std::endl;
    tree_pg_ptr->print_tree();
  }
  UINT itr = 0;
  while (true) {
    if (control.verbose_level >= 2) {
      std::cout << std::endl;
      std::cout << "*********************************************************************************";
      std::cout << "Current iteration = " << itr << std::endl;
      std::cout << "*********************************************************************************";
    }
    if (itr != 0) {
      // no resampling required when itr == 0 since weights haven't been updated yet
      if (control.verbose_level >= 1) {
        std::cout << "iteration = " << itr << ", log p(y|x) = " << log_pd << ", ess/n_particles = " << ess << std::endl;
      }
      particles_ptr = resample(particles_ptr, log_weights_ptr, control, log_pd, ess, weights_norm_ptr, tree_pg_ptr, pgrandom);
    }
    for (UINT i = 0; i < particles_ptr->size(); i++) {
      if (control.verbose_level >= 2) {
        std::cout << "Current particle = " << i << std::endl;
        std::cout << "grow_nodes = " << toString((*particles_ptr)[i]->grow_nodes) << std::endl;
        std::cout << "leaf_nodes = " << toString((*particles_ptr)[i]->tree_ptr->leaf_node_ids) 
          << ", non_leaf_nodes = " << toString((*particles_ptr)[i]->tree_ptr->non_leaf_node_ids) << std::endl;
      }
      if ((*particles_ptr)[i]->grow_nodes.size() > 0) {
        (*particles_ptr)[i]->grow_nodes_itr.push_back((*particles_ptr)[i]->grow_nodes);
      }
      if (tree_pg_ptr->isexisted && i == 0) {
        // the first particle is always bound to the old tree, so it use a special grow function
        grow_next_pg((*particles_ptr)[i], tree_pg_ptr, itr, control);
      }
      else {
        // the function to let the particles grow
        (*particles_ptr)[i]->grow_next(data, param, control, cache, pgrandom);
      }
      (*particles_ptr)[i]->update_depth();
      if (control.verbose_level >= 2) {
        std::cout << "nodes_processed_itr for particle = " << toString((*particles_ptr)[i]->nodes_processed_itr) << std::endl;
        std::cout << "grow_nodes (after running grow_next) (NOT updated for conditioned tree_pg) = " << toString((*particles_ptr)[i]->grow_nodes) << std::endl;
        std::cout << "leaf_nodes = " << toString((*particles_ptr)[i]->tree_ptr->leaf_node_ids) << ", non_leaf_nodes = " 
          << toString((*particles_ptr)[i]->tree_ptr->non_leaf_node_ids) << std::endl; 
        std::cout << "nodes_processed_itr for particle (after running update_particle weights) = " << toString((*particles_ptr)[i]->nodes_processed_itr) << std::endl;
        std::cout << "checking nodes_processed_itr" << std::endl;
      }
    }
    // compute the probability of each particle to be chosen in the next iteratino
    std::tie(log_pd, ess, weights_norm_ptr) = update_particle_weights(particles_ptr, log_weights_ptr, control);     // in place update of log_weights

    if (control.verbose_level >= 2) {
      std::cout<<"log_weights = " << toString(*log_weights_ptr) << std::endl;
    }
    if (check_do_not_grow(particles_ptr)) {
      if (control.verbose_level >= 1) {
        std::cout << "None of the particles can be grown any further; breaking out" << std::endl;
      }
      break;
    }
    itr += 1;
  }

  if (control.if_debug && tree_pg_ptr->isexisted) {
    for (UINT i = 0; i < particles_ptr->size(); i++) {
      if (control.verbose_level >= 2) {
        std::cout << "checking pid = " << i << std::endl;
      }
      (*particles_ptr)[i]->check_nodes_processed_itr();
    }
    if (control.verbose_level >= 2){
      std::cout << "check if tree_pg did the right thing: " << std::endl;
      std::cout << "nodes_processed_itr (orig, new): \n" <<  toString(tree_pg_ptr->nodes_processed_itr) << "\n" 
        << toString((*particles_ptr)[0]->nodes_processed_itr) << std::endl;
      std::cout << "leaf_nodes (orig, new): \n" << toString(tree_pg_ptr->tree_ptr->leaf_node_ids) 
        << toString((*particles_ptr)[0]->tree_ptr->leaf_node_ids) << std::endl;
      std::cout << "non_leaf_nodes (orig, new): \n" << toString(tree_pg_ptr->tree_ptr->non_leaf_node_ids) 
        << toString((*particles_ptr)[0]->tree_ptr->non_leaf_node_ids) << std::endl;
      std::cout << "grow_nodes_itr (orig, new): \n" << toString(tree_pg_ptr->grow_nodes_itr) 
        << toString((*particles_ptr)[0]->grow_nodes_itr) << std::endl;
    }
    if ((*particles_ptr)[0]->tree_ptr->leaf_node_ids != tree_pg_ptr->tree_ptr->leaf_node_ids) {
      throw std::exception();
    }
    if ((*particles_ptr)[0]->tree_ptr->non_leaf_node_ids != tree_pg_ptr->tree_ptr->non_leaf_node_ids) {
      throw std::exception();
    }
    if ((*particles_ptr)[0]->grow_nodes_itr != tree_pg_ptr->grow_nodes_itr) {
      throw std::exception();
    }
  }
  return make_tuple(particles_ptr, ess, log_weights_ptr, log_pd);
}

tuple<Vec_Particle_Ptr, double, DoubleVector_Ptr> init_run_smc(const Data& data, const Control& control, const Param& param, const Cache& cache, const CacheTemp& cache_tmp, Particle_Ptr tree_pg_ptr, Random& pgrandom){

  // create new particles and their weights using the cache_tmp of the last iteration
  Vec_Particle_Ptr particles_ptr;
  DoubleVector_Ptr log_weights_ptr;
  std::tie(particles_ptr, log_weights_ptr) = init_particles(data, control, param, cache_tmp);
  double ess;
  double log_pd;
  std::tie(particles_ptr, ess, log_weights_ptr, log_pd) =
    run_smc(particles_ptr, data, control, param, log_weights_ptr, cache, tree_pg_ptr, pgrandom);
  return make_tuple(particles_ptr, log_pd, log_weights_ptr);
}

bool check_do_not_grow(Vec_Particle_Ptr particles_ptr) {
  bool do_not_grow = true;
  for (Particle_Ptr p_ptr : *particles_ptr) {
    do_not_grow = do_not_grow && p_ptr->do_not_grow;
  }
  return do_not_grow;
}

void grow_next_pg(Particle_Ptr particle_ptr, Particle_Ptr tree_pg_ptr, const UINT& itr, const Control& control) {
  particle_ptr->log_sis_ratio = 0.0;
  particle_ptr->do_not_grow = false;
  particle_ptr->grow_nodes.clear();
  double log_sis_ratio_loglik_old, log_sis_ratio_prior;

  if (itr < tree_pg_ptr->nodes_processed_itr.size()) {
    IntVector& nodes_processed = tree_pg_ptr->nodes_processed_itr[itr];
    particle_ptr->nodes_processed_itr.push_back(nodes_processed);
    for (UINT tmp_id = 0; tmp_id < nodes_processed.size() - 1; tmp_id++) {
      if (!tree_pg_ptr->do_not_split[nodes_processed[tmp_id]]) {
        printf("\nerror with nodes_processed\n");
        exit(1);
      }
      particle_ptr->do_not_split[nodes_processed[tmp_id]] = true;
    }
    UINT node_id = nodes_processed[nodes_processed.size() - 1];
    if (check_if_included(tree_pg_ptr->node_info, node_id)) {
      UINT left = tree_pg_ptr->tree_ptr->getLeftNodeID(node_id);
      UINT right = tree_pg_ptr->tree_ptr->getRightNodeID(node_id);
      double log_sis_ratio_loglik_new = tree_pg_ptr->loglik[left] + tree_pg_ptr->loglik[right] - tree_pg_ptr->loglik[node_id];
      if (check_if_included(tree_pg_ptr->log_sis_ratio_d, node_id)) {
        std::tie(log_sis_ratio_loglik_old, log_sis_ratio_prior) = tree_pg_ptr->log_sis_ratio_d[node_id];
      }
      else {
        printf("\nerror with log_sis_ration_d\n");
        exit(1);
      }
      if (control.verbose_level >= 2) {
        std::cout << "log_sis_ratio_loglik_old = " << log_sis_ratio_loglik_old << std::endl;
        std::cout << "log_sis_ratio_loglik_new = " << log_sis_ratio_loglik_new << std::endl;
      }

      // update log_sis_ratio, leaf_node, non_leaf_node
      particle_ptr->log_sis_ratio = log_sis_ratio_loglik_new + log_sis_ratio_prior;
      tree_pg_ptr->log_sis_ratio_d[node_id] = make_tuple(log_sis_ratio_loglik_new, log_sis_ratio_prior);
      particle_ptr->log_sis_ratio_d[node_id] = tree_pg_ptr->log_sis_ratio_d[node_id];
      particle_ptr->tree_ptr->non_leaf_node_ids.push_back(node_id);
      if (check_if_included(particle_ptr->tree_ptr->leaf_node_ids, node_id)){
        IntVector::iterator it;
        for (it = particle_ptr->tree_ptr->leaf_node_ids.begin(); it != particle_ptr->tree_ptr->leaf_node_ids.end(); it++){
          if (*it == node_id){
            it = particle_ptr->tree_ptr->leaf_node_ids.erase(it);
            break;
          }
        }
      }
      else {
        std::cout << "warning: unable to remove node_id = " << node_id << " from leaf_nodes = " 
          << toString(particle_ptr->tree_ptr->leaf_node_ids) << std::endl;
      }
      // process for the two child node
      particle_ptr->tree_ptr->leaf_node_ids.push_back(left);
      particle_ptr->tree_ptr->leaf_node_ids.push_back(right);
      particle_ptr->node_info[node_id] = tree_pg_ptr->node_info[node_id];
      particle_ptr->logprior[node_id] = tree_pg_ptr->logprior[node_id];
      IntVector tmp{ left, right };
      for (auto node_id_child : tmp) {
        particle_ptr->do_not_split[node_id_child] = false;
        particle_ptr->loglik[node_id_child] = tree_pg_ptr->loglik[node_id_child];
        particle_ptr->logprior[node_id_child] = tree_pg_ptr->logprior[node_id_child];
        particle_ptr->train_ids[node_id_child] = tree_pg_ptr->train_ids[node_id_child];
        particle_ptr->sum_y[node_id_child] = tree_pg_ptr->sum_y[node_id_child];
        particle_ptr->sum_y2[node_id_child] = tree_pg_ptr->sum_y2[node_id_child];
        particle_ptr->mu_mean_post[node_id_child] = tree_pg_ptr->mu_mean_post[node_id_child];
        particle_ptr->mu_prec_post[node_id_child] = tree_pg_ptr->mu_prec_post[node_id_child];
        particle_ptr->n_points[node_id_child] = tree_pg_ptr->n_points[node_id_child];
      }
    }
    if (control.verbose_level >= 2) {
      std::cout << "p.leaf_nodes = " << toString(particle_ptr->tree_ptr->leaf_node_ids) << std::endl;
      std::cout << "p.non_leaf_nodes = " << toString(particle_ptr->tree_ptr->non_leaf_node_ids) << std::endl;
    }
    if (itr + 1 < tree_pg_ptr->grow_nodes_itr.size()) {
      particle_ptr->grow_nodes = tree_pg_ptr->grow_nodes_itr[itr + 1];
      particle_ptr->log_sis_ratio_d = tree_pg_ptr->log_sis_ratio_d;
      particle_ptr->tree_ptr->tree_depth = tree_pg_ptr->tree_ptr->tree_depth;
    }
    else {
      particle_ptr->do_not_grow = true;
    }
  }
  else {
    particle_ptr->do_not_grow = true;
  }
}

IntVector sample_multinomial_numpy(const UINT& n_particles, const DoubleVector& prob, Random& pgrandom){
  IntVector indices = pgrandom.sample_multinomial_distribution(prob, n_particles);
  IntVector pid_list;
  for (UINT i = 0; i < indices.size(); i++) {
    for (UINT j = 0; j < indices[i]; j++) {
      pid_list.push_back(i);
    }
  }
  return pid_list;
}

} // namespace pgbart
