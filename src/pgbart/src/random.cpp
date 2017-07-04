#include <random>
#include <cmath>

#include "pgbart/include/random.hpp"

namespace pgbart {
std::random_device rd;
std::mt19937_64 generator(rd());

void set_random_seed(UINT seed_id) {
  generator.seed(seed_id);
}

double simulate_continuous_uniform_distribution(double minVaule, double maxValue) {
  std::uniform_real_distribution<double> realValue(minVaule, maxValue);
  return realValue(generator);
}

UINT simulate_discrete_uniform_distribution(UINT minValue, UINT maxValue) {
  std::uniform_int_distribution<UINT> intValue(minValue, maxValue);
  return intValue(generator);
}

int ramdom_choice(const IntVector& vec) {
  const int len = vec.size();
  const int id = simulate_discrete_uniform_distribution(0, len - 1);
  return vec[id];
}

double simulate_normal_distribution(double mean, double stddev) {
  std::normal_distribution<> normal_dis(mean, stddev);
  return normal_dis(generator);
}

DoubleVector simulate_normal_distribution(double mean, double stddev, UINT n) {
  DoubleVector temp(n);
  for (UINT i = 0; i < n;  i++)
    temp[i] = simulate_normal_distribution(mean, stddev);
  return temp;
}

double simulate_gamma_distribution(double alpha, double beta) {
  std::gamma_distribution<double> gamma(alpha, beta);
  return gamma(generator);
}



UINT sample_multinomial_distribution(const DoubleVector& probs) {
  double prob = simulate_continuous_uniform_distribution(0.0, 1.0);
  DoubleVector sum = math::cumsum(probs);
  UINT id = 0;

  if (prob < sum[0])
    id = 0;
  else {
    for (size_t i = 1; i < sum.size(); i++) {
      if (prob >= sum[i - 1] && prob < sum[i]) {
        id = i;
        break;
      }
    }
  }
  return id;
}

IntVector sample_multinomial_distribution(const DoubleVector& probs, const UINT& n_time) {
  IntVector temp(probs.size(), 0);
  for (size_t i = 0; i < n_time; i++)
  {
    UINT k = sample_multinomial_distribution(probs);
    temp[k]++;
  }
  return temp;
}

UINT sample_multinomial_scores(const DoubleVector& scores) {
  DoubleVector scoers_cumsum = math::cumsum(scores);
  double s = *(scoers_cumsum.end() - 1) * simulate_continuous_uniform_distribution(0.0, 1.0);
  UINT k = 0;
  BoolVector tmp = math::compare_if(scoers_cumsum, "<", s);
  UINT length = tmp.size();
  for (UINT i = 0; i < length; i++)
    if (tmp[i])
      k++;
  return k;
}

} // namespace pgbart
