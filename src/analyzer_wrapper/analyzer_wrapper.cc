#include "analyzer.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#ifdef _OPENMP
    #include <omp.h>
#else
    #error OpenMP is not available
#endif

#include "mapping_space.h"

namespace py = pybind11;

class AnalyzerWrapper {
public:
    AnalyzerWrapper(const std::string& accelerator_path, const std::string& network_path, unsigned layer_idx) 
        : accelerator(new accelerator_t(accelerator_path)),
          network(new network_t(network_path)),
          analyzer(accelerator, network) {
            network->init_network();
            scheduling_table = scheduling_table_t(accelerator, network);
            scheduling_table.init();
            scheduling_table.load_dnn_layer(layer_idx);
            analyzer.init(scheduling_table);
    }

    ~AnalyzerWrapper() {
        delete accelerator;
        delete network;
    }

    void print_info() {
        std::cout << "Accelerator info:" << std::endl;
        accelerator->print_spec();
        std::cout << "Network info:" << std::endl;
        network->print_stats();
    }

    void init(const std::string& scheduling_table_path) {
        scheduling_table = scheduling_table_t(accelerator, network, scheduling_table_path);
        analyzer.init(scheduling_table);
    }

    std::vector<std::vector<unsigned>> get_scheduling_table() {
        std::vector<std::vector<unsigned>> table_values;
        for (unsigned i = 0; i < scheduling_table.get_num_rows(); ++i) {
            table_values.push_back(scheduling_table.get_row_values(i));
        }
        return table_values;
    }

    std::vector<unsigned> get_layer_parameters(unsigned idx) {
        return network->get_layer_parameters(idx);
    }

    std::vector<bool> get_fixed_rows() {
        std::vector<bool> fixed_rows;
        for (unsigned i = 0; i < scheduling_table.get_num_rows(); ++i) {
            fixed_rows.push_back(scheduling_table.is_skippable(i));
        }
        return fixed_rows;
    }

    std::vector<std::vector<unsigned>> search_optimized_table(std::vector<unsigned>& rows_idx, unsigned rows_num, std::vector<unsigned>& products, unsigned metric) {
        // Assert that the number of row indices matches the specified number of rows
        assert(rows_idx.size() == rows_num && "Number of row indices must match the specified number of rows");

        mapping_space_t mapping_space;
        mapping_space.generate(rows_num, products);
        // Debugging: Print the mapping space
        // mapping_space.print_permutations();
        // std::cout << "Number of permutations: " << mapping_space.get_num_permutations() << std::endl;

        scheduling_table_t best_scheduling_table;
        float best_energy = std::numeric_limits<float>::infinity();
        float best_cycle = std::numeric_limits<float>::infinity();

        #pragma omp parallel
        {
            scheduling_table_t local_best_scheduling_table;
            float local_best_energy = std::numeric_limits<float>::infinity();
            float local_best_cycle = std::numeric_limits<float>::infinity();
            analyzer_t local_analyzer(accelerator, network);

            #pragma omp for nowait
            for (unsigned i = 0; i < mapping_space.get_num_permutations(); ++i) {
                std::vector<std::vector<unsigned>> mapping_values_set = mapping_space.get_mapping_set(i);

                scheduling_table_t scheduling_table_curr = scheduling_table;
                for (unsigned j = 0; j < rows_num; ++j) {
                    unsigned row_idx = rows_idx[j];
                    scheduling_table_curr.update_row(row_idx, mapping_values_set[j]);
                }

                local_analyzer.init(scheduling_table_curr);
                if (!local_analyzer.check_validity()) continue;

                local_analyzer.estimate_cost();
                float curr_energy = local_analyzer.get_total_cost(metric_t::ENERGY);
                float curr_cycle = local_analyzer.get_total_cost(metric_t::CYCLE);

                if ((metric_t)metric == metric_t::ENERGY) {
                    if (curr_energy < local_best_energy) {
                        local_best_energy = curr_energy;
                        local_best_scheduling_table = scheduling_table_curr;
                    }
                }
                else if ((metric_t)metric == metric_t::CYCLE) {
                    if (curr_cycle < local_best_cycle) {
                        local_best_cycle = curr_cycle;
                        local_best_scheduling_table = scheduling_table_curr;
                    }
                }
            }

            #pragma omp critical
            {
                if ((metric_t)metric == metric_t::ENERGY) {
                    if (local_best_energy < best_energy) {
                        best_energy = local_best_energy;
                        best_scheduling_table = local_best_scheduling_table;
                    }
                }
                else if ((metric_t)metric == metric_t::CYCLE) {
                    if (local_best_cycle < best_cycle) {
                        best_cycle = local_best_cycle;
                        best_scheduling_table = local_best_scheduling_table;
                    }
                }
            }
        }

        analyzer.init(best_scheduling_table);
        scheduling_table = best_scheduling_table;

        return get_scheduling_table();
    }

    std::vector<std::vector<unsigned>> search_optimized_table_sequence(std::vector<unsigned>& rows_idx, unsigned rows_num, std::vector<unsigned>& products, unsigned metric) {
        // Assert that the number of row indices matches the specified number of rows
        assert(rows_idx.size() == rows_num && "Number of row indices must match the specified number of rows");
        
        mapping_space_t mapping_space;
        std::vector<std::vector<unsigned>> mapping_values_set;
        
        scheduling_table_t best_scheduling_table;
        float best_energy = std::numeric_limits<float>::infinity();
        float best_cycle = std::numeric_limits<float>::infinity();
        
        mapping_space.generate(rows_num, products);
        
        // Traverse all possible mapping options in targeted levels
        while(!mapping_space.is_last()) {
            // Get new mapping values
            mapping_values_set = mapping_space.get_mapping_set();

            // Update mapping values of scheduling table
            // Create a copy of the current scheduling table
            scheduling_table_t scheduling_table_curr = scheduling_table;
            // Update the rows of the scheduling table
            for (unsigned i = 0; i < rows_num; ++i) {
                unsigned row_idx = rows_idx[i];
                scheduling_table_curr.update_row(row_idx, mapping_values_set[i]);
            }

            // Load scheduling table to analyzer
            analyzer.init(scheduling_table_curr);
            // Check_Validity
            if(!analyzer.check_validity()) continue;

            analyzer.estimate_cost();
            float curr_energy = analyzer.get_total_cost(metric_t::ENERGY);
            float curr_cycle = analyzer.get_total_cost(metric_t::CYCLE);

            if ((metric_t)metric == metric_t::ENERGY) {
                if (curr_energy < best_energy) {
                    best_energy = curr_energy;
                    best_scheduling_table = scheduling_table_curr;
                }
            }
            else if ((metric_t)metric == metric_t::CYCLE) {
            if (curr_cycle < best_cycle) {
                    best_cycle = curr_cycle;
                    best_scheduling_table = scheduling_table_curr;
                }
            }
        }
        
        analyzer.init(best_scheduling_table);
        scheduling_table = best_scheduling_table;
        
        return get_scheduling_table();
    }

    void update_scheduling_table(const std::vector<std::vector<unsigned>>& new_scheduling_table) {
        for (unsigned i = 0; i < new_scheduling_table.size(); ++i) {
            scheduling_table.update_row(i, new_scheduling_table[i]);
        }
        analyzer.init(scheduling_table);
    }

    void estimate_cost() {
        analyzer.estimate_cost();
    }

    float get_total_energy() {
        return analyzer.get_total_cost(metric_t::ENERGY);
    }

    float get_total_cycle() {
        return analyzer.get_total_cost(metric_t::CYCLE);
    }

private:
    accelerator_t *accelerator;
    network_t *network;
    analyzer_t analyzer;
    scheduling_table_t scheduling_table;
};

PYBIND11_MODULE(analyzer_wrapper, m) {
    py::class_<AnalyzerWrapper>(m, "Analyzer")
        .def(py::init<const std::string&, const std::string&, unsigned>())
        .def("print_info", &AnalyzerWrapper::print_info)
        .def("init", &AnalyzerWrapper::init)
        .def("get_scheduling_table", &AnalyzerWrapper::get_scheduling_table)
        .def("get_layer_parameters", &AnalyzerWrapper::get_layer_parameters)
        .def("get_fixed_rows", &AnalyzerWrapper::get_fixed_rows)
        .def("search_optimized_table", &AnalyzerWrapper::search_optimized_table)
        .def("search_optimized_table_sequence", &AnalyzerWrapper::search_optimized_table_sequence)
        .def("update_scheduling_table", &AnalyzerWrapper::update_scheduling_table)
        .def("estimate_cost", &AnalyzerWrapper::estimate_cost)
        .def("get_total_energy", &AnalyzerWrapper::get_total_energy)
        .def("get_total_cycle", &AnalyzerWrapper::get_total_cycle);
}
