#ifndef PGBART_CONTROL_HPP
#define PGBART_CONTROL_HPP

#include <pgbart/include/config.hpp>
#include <pgbart/include/math.hpp>
#include <fstream>

/**************************************
File name : control.hpp
Date : 2016-12-7
Struct List : Control
***************************************/

namespace pgbart
{
    struct Control
    {
        // See the meaning of the vars in interface.cpp
        double alpha_bart;
        double alpha_split;
        double beta_split;
        bool if_center_label;
        bool if_debug;
        double ess_threshold;
        UINT init_seed_id;
        bool if_set_seed;
        double k_bart;
        UINT m_bart;
        UINT min_size;
        UINT ndpost;
        UINT nskip;
        UINT keepevery;
        string variance_type;
        double q_bart;
        double lambda_bart;
        UINT verbose_level;
        UINT n_particles;
        string resample_type;


        Control() : alpha_bart(3.0), alpha_split(0.95), beta_split(0.5), if_center_label(false), if_debug(false), ess_threshold(1.0), init_seed_id(1),
            if_set_seed(false), k_bart(2.0), m_bart(1), min_size(1), ndpost(1000), nskip(100), keepevery(1), variance_type("unconditional"), q_bart(0.9), lambda_bart(0.5),
            verbose_level(0), n_particles(10), resample_type("multinomial"){}

        Control(double alpha_bart, double alpha_split, double beta_split, bool if_center_label, bool if_debug, double ess_threshold, UINT init_seed_id,
            bool if_set_seed, double k_bart, UINT m_bart, UINT min_size, UINT ndpost, UINT nskip, UINT keepevery, string variance_type, double q_bart,
            UINT verbose_level, UINT n_particles, string resample_type) {
            this->alpha_bart = alpha_bart;
            this->alpha_split = alpha_split;
            this->beta_split = beta_split;
            this->if_center_label = if_center_label;
            this->if_debug = if_debug;
            this->ess_threshold = ess_threshold;
            this->init_seed_id = init_seed_id;
            this->if_set_seed = if_set_seed;
            this->k_bart = k_bart;
            this->m_bart = m_bart;
            this->min_size = min_size;
            this->ndpost = ndpost;
            this->nskip = nskip;
            this->keepevery = keepevery;
            this->variance_type = variance_type;
            this->q_bart = q_bart;
            this->verbose_level = verbose_level;
            this->n_particles = n_particles;
            this->resample_type = resample_type;
        }

        string toString() {
            std::ostringstream os;
            os << "alpha_bart = " << this->alpha_bart << "\n";
            os << "alpha_split = " << this->alpha_split << "\n";
            os << "beta_split = " << this->beta_split << "\n";
            os << "if_center_label = " << this->if_center_label << "\n";
            os << "if_debug = " << this->if_debug << "\n";
            os << "ess_threshold = " << this->ess_threshold << "\n";
            os << "init_seed_id = " << this->init_seed_id << "\n";
            os << "if_set_seed = " << this->if_set_seed << "\n";
            os << "k_bart = " << this->k_bart << "\n";
            os << "m_bart = " << this->m_bart << "\n";
            os << "min_size = " << this->min_size << "\n";
            os << "ndpost = " << this->ndpost << "\n";
            os << "nskip = " << this->nskip << "\n";
            os << "keepevery = " << this->keepevery << "\n";
            os << "variance_type = " << this->variance_type << "\n";
            os << "q_bart = " << this->q_bart << "\n";
            os << "lambda_bart = " << this->lambda_bart << "\n";
            os << "verbose_level = " << this->verbose_level << "\n";
            os << "n_particles = " << this->n_particles << "\n";
            os << "resample_type = " << this->resample_type << "\n";
            return os.str();
        }
    };
}

#endif
