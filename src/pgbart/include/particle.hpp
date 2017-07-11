#ifndef PGBART_PARTICLE_HPP
#define PGBART_PARTICLE_HPP

#include <cstring>
#include <memory>
#include <vector>

#include "pgbart/include/state.hpp"

namespace pgbart {
  
class Particle;
typedef std::shared_ptr<Particle> Particle_Ptr;
typedef std::shared_ptr<vector<Particle_Ptr>> Vec_Particle_Ptr;

class Particle : public State {
public:
  IntVector grow_nodes; // nodes can grow
  vector<IntVector> grow_nodes_itr; // nodes have grown 
  IntVector ancestry; // ancestral particle id
  bool do_not_grow; // whether the particle can grow
  double log_sis_ratio;
  vector<IntVector> nodes_processed_itr;// the nodes processed
  double pred_val_logprior;
  bool isexisted;

public:

  Particle(const IntVector& train_ids, const Param& param, const CacheTemp& cache_temp);
  Particle() { this->isexisted = true; };
  Particle(const Particle& p);
  
  double process_node_id(const Data& data_train, const Param& param, const Control& control, const Cache& cache, const UINT& node_id, Random& pgrandom);

  void grow_next(const Data& data_train, const Param& param, const Control& control, const Cache& cache, Random& pgrandom);

  void check_nodes_processed_itr();

  
};

} // namespace pgbart

// *****************************************************************************************************
// *****************************************************************************************************

namespace pgbart {

/*
return {"double log_pd", "double ess", "DoubleVector log_weights_new", "DoubleVector weights_norm_new"}
*/
tuple<double, double, DoubleVector_Ptr>
  update_particle_weights(Vec_Particle_Ptr particles_ptr, DoubleVector_Ptr log_weights_ptr, const Control& control);

/* return {"Particle op", "DoubleVector log_weights_new"} */
Vec_Particle_Ptr resample(const Vec_Particle_Ptr particles_ptr, DoubleVector_Ptr log_weights_ptr, const Control& control,
  const double& log_pd, const double& ess, DoubleVector_Ptr weights_norm_ptr, Particle_Ptr tree_pg_ptr, Random& pgrandom);

IntVector_Ptr resample_pids_basic(const Control& control, const UINT& n_particles, DoubleVector_Ptr probs, Random& pgrandom);

IntVector_Ptr sample_multinomial_particle(const UINT& n_particles, DoubleVector_Ptr probs_ptr, Random& pgrandom);

IntVector_Ptr sample_systematic_particle(const UINT& n_particles, DoubleVector_Ptr probs_ptr, Random& pgrandom);

Vec_Particle_Ptr create_new_particles(const Vec_Particle_Ptr particles_ptr, IntVector_Ptr pid_list_ptr, const Control& control);

Particle_Ptr copy_particle(Particle_Ptr particle_ptr);

tuple<Vec_Particle_Ptr, DoubleVector_Ptr>
  init_particles(const Data& data_train, const Control& control, const Param& param, const CacheTemp cache_temp);

tuple<Vec_Particle_Ptr, double, DoubleVector_Ptr, double>
  run_smc(Vec_Particle_Ptr particles_ptr, const Data& data, const Control& control, const Param& param,
  DoubleVector_Ptr log_weights_ptr, const Cache& cache, Particle_Ptr tree_pg_ptr, Random& pgrandom);

tuple<Vec_Particle_Ptr, double, DoubleVector_Ptr>
  init_run_smc(const Data& data, const Control& control, const Param& param, const Cache& cache, 
  const CacheTemp& cache_tmp, Particle_Ptr tree_pg_ptr, Random& pgrandom);

bool check_do_not_grow(Vec_Particle_Ptr particles_ptr);

void grow_next_pg(Particle_Ptr particle_ptr, Particle_Ptr tree_pg_ptr, const UINT& itr, const Control& control);


IntVector sample_multinomial_numpy(const UINT& n_particles, const DoubleVector& prob, Random& pgrandom);

} // namesapce pgbart

#endif