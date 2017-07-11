#ifndef PGBART_RANDOM_HPP
#define PGBART_RANDOM_HPP

//#define USE_PYTHON

#include "pgbart/include/config.hpp"
#include "pgbart/include/math.hpp"

/**************************************
File name : random.hpp
Date : 2016-12-7
Function List : void set_random_seed(UINT seed_id)
        simulate_continuous_uniform_distribution(double minVaule, double maxValue)
        simulate_normal_distribution(double mean, double stddev)
        simulate_normal_distribution(double mean, double stddev, UINT n)
        simulate_gamma_distribution(double alpha, double beta)
        sample_multinomial_distribution(const DoubleVector& probs, const UINT& n_time)
***************************************/

namespace pgbart {
  
  class Random {
  public:
    std::random_device rd;
    std::mt19937 generator;
    std::default_random_engine shuffle_gen;

  public:
    Random();
    void set_random_seed(UINT seed_id);

    // generte double form a half-closed interval [minValue, maxValue)
    double simulate_continuous_uniform_distribution(double minVaule, double maxValue);
    // generte integer form a closed interval [minValue, maxValue]
    UINT simulate_discrete_uniform_distribution(UINT minValue, UINT maxValue);
    int ramdom_choice(const IntVector& vec);

    //double simulateNormalDistribution(double mean, double variance);
    double simulate_normal_distribution(double mean, double stddev);

    DoubleVector simulate_normal_distribution(double mean, double stddev, UINT n);

    double simulate_gamma_distribution(double alpha, double beta);

    UINT sample_multinomial_distribution(const DoubleVector& probs);

    IntVector sample_multinomial_distribution(const DoubleVector& probs, const UINT& n_time);

    UINT sample_multinomial_scores(const DoubleVector& scores);

    void shuffle(IntVector& ori_order);

  };
} // namespace pgbart

#endif