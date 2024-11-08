#ifndef __MAPPING_SPACE_H__
#define __MAPPING_SPACE_H__

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "enums.h"

/* Mapping space */
class mapping_space_t {
public:
    mapping_space_t();
    mapping_space_t(unsigned num_levels_,
                    uint64_t num_permutations_,
                    std::vector<std::vector<std::vector<unsigned>>> layer_permutations_);
    mapping_space_t(mapping_space_t& mapping_space_);
    ~mapping_space_t();
    void clear();
    void print_permutations() const;
    void generate(const unsigned num_levels_, 
                  const std::vector<unsigned> &layer_values_); 
    bool is_last() const;
    std::vector<std::vector<unsigned>> get_mapping_set();
    std::vector<std::vector<unsigned>> get_mapping_set(unsigned index);
    mapping_space_t partition_off(unsigned tid_, 
                                  unsigned num_threads_) const;
    uint64_t get_num_permutations() const;
    std::vector<std::vector<std::vector<unsigned>>> get_layer_permutations() const;

private:
    std::vector<unsigned> get_factors(const unsigned val_);
    std::vector<std::vector<unsigned>> get_permutations(const unsigned idx_) const;
    void get_permutations(const unsigned idx_, 
                          const unsigned val_, 
                          std::vector<unsigned> &permutation_);
    // Variables & containers
    unsigned num_levels;
    uint64_t num_permutations;
    std::vector<std::vector<std::vector<unsigned>>> layer_permutations;
    
    uint64_t permutation_index;
    std::vector<uint64_t> parameter_index;
};

/* Mapping space range per thread */
class range_t {
public:
    range_t(const unsigned tid_, 
            const unsigned num_threads_,
            const std::vector<std::vector<std::vector<unsigned>>>& layer_permutations_);
    ~range_t();
    
    size_t start_k;
    size_t end_k;
    size_t start_b;
    size_t end_b;
    size_t start_p;
    size_t end_p;
    size_t start_q;
    size_t end_q;
    size_t start_c;
    size_t end_c;
    size_t start_r;
    size_t end_r;
    size_t start_s;
    size_t end_s;
    size_t start_g;
    size_t end_g;
};

#endif
