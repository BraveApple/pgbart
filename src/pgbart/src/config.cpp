#include <sstream>

#include "pgbart/include/config.hpp"
#include "pgbart/include/math.hpp"

namespace pgbart {

string SplitInfo::toString() {
  std::ostringstream os;
  os << "feat_id_chosen = " << feat_id_chosen << " ";
  os << "split_chosen = " << split_chosen << " ";
  os << "idx_split_global = " << idx_split_global;
  return os.str();
}

string DimensionInfo::toString() {
  std::ostringstream os;
  os << "x_min = " << x_min << " ";
  os << "x_max = " << x_max << " ";
  os << "idx_min = " << idx_min << " ";
  os << "idx_max = " << idx_max << " ";
  os << "\n";
  os << "feat_score_cumsum_prior_current = \n";
  os << math::toString(feat_score_cumsum_prior_current);
  return os.str();
}

string Param::toString() {
  std::ostringstream os;
  os << "alpha_bart = " << alpha_bart << " ";
  os << "alpha_split = " << alpha_split << " ";
  os << "beta_bart = " << beta_bart << " ";
  os << "beta_split = " << beta_split << " ";
  os << "k_bart = " << k_bart << " ";
  os << "\n";
  os << "lambda_bart = " << lambda_bart << " ";
  os << "log_lambda_bart = " << log_lambda_bart << " ";
  os << "m_bart = " << m_bart << " ";
  os << "mu_mean = " << mu_mean << " ";
  os << "mu_prec = " << mu_prec;
  return os.str();
}

string CacheTemp::toString() {
  std::ostringstream os;
  os << "isexisted = " << isexisted << "\n";
  // parent node
  os << "n_points = " << n_points << " ";
  os << "sum_y = " << sum_y << " ";
  os << "sum_y2 = " << sum_y2 << " ";
  os << "mu_mean_post = " << mu_mean_post << " ";
  os << "mu_prec_post = " << mu_prec_post << " ";
  os << "loglik = " << loglik << " ";
  os << "\n";
  // left child node
  os << "n_points_left = " << n_points_left << " ";
  os << "sum_y_left = " << sum_y_left << " ";
  os << "sum_y2_left = " << sum_y2_left << " ";
  os << "mu_mean_post_left = " << mu_mean_post_left << " ";
  os << "mu_prec_post_left = " << mu_prec_post_left << " ";
  os << "loglik_left = " << loglik_left << " ";
  os << "\n";
  // right child node
  os << "n_points_right = " << n_points_right << " ";
  os << "sum_y_right = " << sum_y_right << " ";
  os << "sum_y2_right = " << sum_y2_right << " ";
  os << "mu_mean_post_right = " << mu_mean_post_right << " ";
  os << "mu_prec_post_right = " << mu_prec_post_right << " ";
  os << "loglik_right = " << loglik_right;
  return os.str();
}

} // namespace pgbart
