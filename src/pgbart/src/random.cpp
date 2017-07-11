#include <random>
#include <cmath>
#include <iostream>

#include "pgbart/include/random.hpp"

namespace pgbart {

  Random::Random() {}

void Random::set_random_seed(UINT seed_id) {
  generator.seed(seed_id);
  shuffle_gen.seed(seed_id);
}

double Random::simulate_continuous_uniform_distribution(double minVaule, double maxValue) {
  std::uniform_real_distribution<double> realValue(minVaule, maxValue);
  return realValue(generator);
}

UINT Random::simulate_discrete_uniform_distribution(UINT minValue, UINT maxValue) {
  std::uniform_int_distribution<UINT> intValue(minValue, maxValue);
  return intValue(generator);
}

int Random::ramdom_choice(const IntVector& vec) {
  const int len = vec.size();
  const int id = simulate_discrete_uniform_distribution(0, len - 1);
  return vec[id];
}

double Random::simulate_normal_distribution(double mean, double stddev) {
  std::normal_distribution<> normal_dis(mean, stddev);
  return normal_dis(generator);
}

DoubleVector Random::simulate_normal_distribution(double mean, double stddev, UINT n) {
  DoubleVector temp(n);
  for (UINT i = 0; i < n;  i++)
    temp[i] = simulate_normal_distribution(mean, stddev);
  return temp;
}

double Random::simulate_gamma_distribution(double alpha, double beta) {
  std::gamma_distribution<double> gamma(alpha, beta);
  return gamma(generator);
}


UINT Random::sample_multinomial_distribution(const DoubleVector& probs) {
  double prob = this->simulate_continuous_uniform_distribution(0.0, 1.0);
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

IntVector Random::sample_multinomial_distribution(const DoubleVector& probs, const UINT& n_time) {
  IntVector temp(probs.size(), 0);
  for (size_t i = 0; i < n_time; i++)
  {
    UINT k = sample_multinomial_distribution(probs);
    temp[k]++;
  }
  return temp;
}

UINT Random::sample_multinomial_scores(const DoubleVector& scores) {
  DoubleVector scoers_cumsum = math::cumsum(scores);
  double s = *(scoers_cumsum.end() - 1) * this->simulate_continuous_uniform_distribution(0.0, 1.0);
  UINT k = 0;
  BoolVector tmp = math::compare_if(scoers_cumsum, "<", s);
  UINT length = tmp.size();
  for (UINT i = 0; i < length; i++)
    if (tmp[i])
      k++;
  return k;
}

void Random::shuffle(IntVector& ori_order) {
  std::shuffle(ori_order.begin(), ori_order.end(), this->shuffle_gen);
}

} // namespace pgbart
