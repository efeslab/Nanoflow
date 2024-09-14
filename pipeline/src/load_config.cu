#include "config.h"
#include <string>
#include <nlohmann/json.hpp>
#include <fstream>

vortexModelConfig ModelConfig;

using json = nlohmann::json;


void updateVortexModelConfig(const std::string& filename) {
    std::ifstream file(filename); 
	if (!file.is_open()) { 
		std::cerr << "Could not open the file: " << filename << std::endl;
		return; 
	} 
    // Read the file contents into a string 
	json j; 
	file >> j; 
	// Close the file 
	file.close();
    // Parse the JSON string
    // Extract values from the "model" section
    int hidden_size = j["model"]["hidden_size"].get<int>();
    int num_attention_heads = j["model"]["num_attention_heads"].get<int>();
    int num_key_value_heads = j["model"]["num_key_value_heads"].get<int>();
    int intermediate_size = j["model"]["intermediate_size"].get<int>();
    int num_hidden_layers = j["model"]["num_hidden_layers"].get<int>();
    float rms_norm_eps = j["model"]["rms_norm_eps"].get<float>();
    int vocab_size = j["model"]["vocab_size"].get<int>();
    float rope_theta = j["model"].value("rope_theta", 10000.0);
    bool rope_scaling = false; 
    float factor;
    float low_freq_factor;
    float high_freq_factor;
    int original_max_position_embeddings;
    if (j["model"]["rope_scaling"].is_null()) {
        std::cout << "\"rope_scaling\" is null" << std::endl;
    } else {
        std::cout << "\"rope_scaling\" has a value" << std::endl;
        rope_scaling = true;
        factor = j["model"]["rope_scaling"]["factor"].get<float>();
        low_freq_factor = j["model"]["rope_scaling"]["low_freq_factor"].get<float>();
        high_freq_factor = j["model"]["rope_scaling"]["high_freq_factor"].get<float>();
        original_max_position_embeddings = j["model"]["rope_scaling"]["original_max_position_embeddings"].get<int>();
    }
    // Extract or set default values from the "model_configs" section
    int gpu_num = j["model_configs"].value("gpu_num", 8);
    int run_layer = j["model_configs"].value("run_layer", num_hidden_layers);
    int allocate_kv_data_batch = j["model_configs"].value("allocate_kv_data_batch", 400);
    int frame_page_size = j["model_configs"].value("frame_page_size", 16);
    int max_batch_size = j["model_configs"].value("max_batch_size", 2048);
    // Calculate derived values
    int model_head_dim = hidden_size / num_attention_heads;
    int model_gqa = num_attention_heads / num_key_value_heads;
    int model_ff_dim = intermediate_size;
    // Update the global ModelConfig using the extracted and calculated values
    ModelConfig.printConfig();
    if (!rope_scaling) {
        ModelConfig.setConfig(
            gpu_num,
            num_hidden_layers,
            run_layer,
            allocate_kv_data_batch,
            model_head_dim,
            model_gqa,
            num_key_value_heads,
            num_attention_heads,
            model_ff_dim,
            hidden_size,
            frame_page_size,
            max_batch_size,
            rms_norm_eps,
            rope_theta,
            vocab_size,
            1,
            0,
            0,
            0,
            0,
            0,
            false
        );
    } else {
        ModelConfig.setConfig(
            gpu_num,
            num_hidden_layers,
            run_layer,
            allocate_kv_data_batch,
            model_head_dim,
            model_gqa,
            num_key_value_heads,
            num_attention_heads,
            model_ff_dim,
            hidden_size,
            frame_page_size,
            max_batch_size,
            rms_norm_eps,
            rope_theta,
            vocab_size,
            factor,
            low_freq_factor,
            high_freq_factor,
            original_max_position_embeddings,
            0,
            0,
            true
        );
    }
    // The calculateConfig function is called within setConfig to handle derived values.
    ModelConfig.printConfig();
}