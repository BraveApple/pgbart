#ifndef PGBART_PMCMC_HPP
#define PGBART_PMCMC_HPP

#include <map>

#include "pgbart/include/data.hpp"
#include "pgbart/include/control.hpp"
#include "pgbart/include/config.hpp"
#include "pgbart/include/particle.hpp"
#include "pgbart/include/compare.hpp"


namespace pgbart {
class Pmcmc;
typedef std::shared_ptr<Pmcmc> Pmcmc_Ptr;

class Pmcmc {
private:
  double log_pd;
  
public:
  Particle_Ptr p_ptr;
  Pmcmc();
  Pmcmc(const Data& data, const Control& control, const Param& param, const Cache& cache, const CacheTemp& cache_temp);
  bool update_p(Vec_Particle_Ptr particles_ptr, DoubleVector_Ptr log_weights_ptr, double& log_pd, const Control& control);
  bool sample(const Data& data, const Control& control, const Param& param, const Cache& cache, const CacheTemp& cache_tmp);
};

} // namesapce pgbart

namespace pgbart{
Pmcmc_Ptr init_tree_mcmc(const Data& data_train, const Control& control, const Param& param,
  const Cache& cache, const CacheTemp& cache_temp);
tuple<Particle_Ptr, bool> run_mcmc_single_tree(Particle_Ptr p_ptr, const Control& control, const Data& data_train,
  const Param& param, const Cache& cache, bool change, const CacheTemp& cache_temp, Pmcmc_Ptr pmcmc_ptr);
} // namespace pgbart

namespace pgbart {
class TreeMCMC: public State {
private:
  TupleVector inner_pc_paires;
  IntVector both_children_terminal;

  map<UINT, SplitInfo> node_info_new;

public:
  IntVector_Ptr get_nodes_not_in_subtree(const int node_id);

  IntVector_Ptr get_nodes_subtree(const int node_id);

  double log_acc compute_log_acc_g(const int node_id, const Param& param, const int len_both_children_terminal, 
    const double loglik, const IntVector& train_ids_left, const IntVector& train_ids_right, 
    const Cache& cache, const Control& control, const Data& data, const IntVector& grow_nodes);

  double compute_log_inv_acc_p(const int node_id, const Param& param, const int len_both_children_terminal,
    const double loglik, const IntVector& grow_nodes, const Cache& cache, const Control& control, 
    const Data& train_data);

  bool grow(const Data& train_data, const Control& control, const Param& param, const Cache& cache,
    const IntVector& grow_nodes);

  bool prune(const Data& train_data, const Control& control, const Param& param, const Cache& cache,
    const IntVector& grow_nodes);

  bool change(const Data& train_data, const Control& control, const Param& param, const Cache& cache,
      const IntVector& grow_nodes);

  bool swap(const Data& train_data, const Control& control, const Param& param, const Cache& cache,
      const IntVector& grow_nodes);

  tuple<bool, MoveType> sample(const Data& train_data, const Control& control, const Param& param, 
    const Cache& cache);
};

} // namesapce pgbart

#endif