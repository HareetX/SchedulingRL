#include "parser.h"

// Configuration section
section_config_t::section_config_t(std::string name_):
    name(name_) {
}

section_config_t::~section_config_t() {
    settings.clear();
}

// Show all setting values (key, value) pairs
void section_config_t::show_setting() {
    for(auto it = settings.begin(); it != settings.end(); it++) {
        std::cerr << it->first << "=" << it->second << std::endl;
    }
}
// Get total number of settings
unsigned section_config_t::get_num_settings() {
    return settings.size();
}
// Get setting value with settings order
std::string section_config_t::get_value(unsigned idx_) {
    assert(idx_ < settings_order.size());
    return settings_order.at(idx_);
}
// Get setting value with key value
std::string section_config_t::get_value(std::string key_) {
    // Find position of key
    auto it = settings.find(key_);
    if(it != settings.end()) {
        return it->second;
    }
    else {
        std::cerr << "[Error] key(" << key_ << ") does not exist" << std::endl;
        return "False";
    }
}
// Get setting value
std::string section_config_t::get_value(std::string value_, unsigned idx_) {
    assert(idx_ < settings.size());
    unsigned cnt = 0;
    for(auto it = settings.begin(); it != settings.end(); it++) {
        if(cnt == idx_) { 
            if(value_ == "key") { return it->first; }
            else if(value_ == "value") { return it->second; }
            else { std::cerr << "Error" << std::endl; }
            break; 
        }
        cnt++;
        
    }
    return "False";
}
// Add (key, value) pair to the section settings
void section_config_t::add_setting(std::string key_, std::string value_) {
    settings.insert(std::pair<std::string,std::string>(lowercase(key_), lowercase(value_)));
    settings_order.push_back(lowercase(key_));
}
// Check if a setting exists.
bool section_config_t::exist(std::string key_) {
    return settings.find(lowercase(key_)) != settings.end();
} 

// Parser for config files
parser_t::parser_t() { }
parser_t::~parser_t() { 
    sections.clear();
}

void parser_t::cfgparse(const std::string cfg_path_) {
    std::fstream file_stream;
    file_stream.open(cfg_path_.c_str(), std::fstream::in);
    if(!file_stream.is_open()) {
        std::cerr << "Error: failed to open " << cfg_path_ << std::endl;
        exit(1);
    }

    std::string line;
    while(getline(file_stream, line)) {
        // Erase all spaces
        line.erase(remove(line.begin(), line.end(), ' '), line.end());
        // Erase carriage return
        line.erase(remove(line.begin(), line.end(), '\r'), line.end()); 
        // Skip blank lines or comments
        if(!line.size() || (line[0] == '#')) continue;
        // Beginning of [component]
        if(line[0] == '[') {
            std::string section_name = line.substr(1, line.size()-2);
            sections.push_back(section_config_t(section_name));
        }
        else {
            size_t eq = line.find('=');
            if(eq == std::string::npos) {
                std::cerr << "Error: invalid config" << std::endl << line << std::endl;
                exit(1);
            }
            // Save (key, value) pair in the latest section setting
            std::string key   = line.substr(0, eq);
            std::string value = line.substr(eq+1, line.size()-1);
            sections[sections.size()-1].add_setting(key, value);
        }
    }
}
