#include "computeBound.cuh"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "tensorManager.cuh"

static int python_nranks;
static int python_vnranks;

void setRank(int nranks, int vnranks){
    python_nranks = nranks;
    python_vnranks = vnranks;
}

void setKVdata(vortexInitData &inputData, pybind11::array_t<size_t> kv_data){
    pybind11::buffer_info kv_data_info = kv_data.request();
    int *kv_data_ptr_int = static_cast<int *>(kv_data_info.ptr);
    half** kv_data_ptr = reinterpret_cast<half**>(kv_data_ptr_int);
    
    half ** new_data = new half*[kv_data_info.shape[0]];
    for (int i = 0; i < kv_data_info.shape[0]; i++){
        new_data[i] = kv_data_ptr[i];
    }
    inputData.kv_data = new_data;
}

void setWeight(vortexInitData &inputData, vortexModelWeight weight, size_t weight_size){
    inputData.weight = weight;
    inputData.weight_size = weight_size;
}

void setInputEmbeddingUpdate(vortexUpdateData &updateData, size_t input_tokens){
    updateData.input_tokens = reinterpret_cast<int*>(input_tokens);
}


void setKVIndicesUpdate(vortexUpdateData &updateData, size_t kv_indices){
    updateData.kv_indices = reinterpret_cast<int*>(kv_indices);
}


void setKVIndptrUpdate(vortexUpdateData &updateData, size_t kv_indptr){
    updateData.kv_indptr = reinterpret_cast<int*>(kv_indptr);
}

void setKVLastPageLenUpdate(vortexUpdateData &updateData, size_t kv_last_page_len){
    updateData.kv_last_page_len = reinterpret_cast<int*>(kv_last_page_len);
}

void setInputIndptrUpdate(vortexUpdateData &updateData, size_t input_indptr){
    updateData.input_indptr = reinterpret_cast<int*>(input_indptr);
}

void setRevInputIndptrUpdate(vortexUpdateData &updateData, size_t rev_input_indptr){
    updateData.rev_input_indptr = reinterpret_cast<int*>(rev_input_indptr);
}

void setPerTokenOffsetUpdate(vortexUpdateData &updateData, size_t per_token_offset){
    updateData.per_token_offset = reinterpret_cast<int*>(per_token_offset);
}

void setKeepTokenListUpdate(vortexUpdateData &updateData, size_t keep_token_list){
    updateData.keep_token_list = reinterpret_cast<int*>(keep_token_list);
}

void setTmpBuffer(vortexInitData &inputData, size_t tmp_buffer, size_t tmp_buffer_size){
    inputData.tmp_buffer = reinterpret_cast<half*>(tmp_buffer);
    inputData.tmp_buffer_size = tmp_buffer_size;
}

std::vector<vortexOutputData> output;

void pllm_init(std::vector<vortexInitData>& input, Worker::PipelineType pipeTy){
    output.resize(python_nranks);
        // Create a vector to hold the threads
    std::vector<std::thread> threads;

    // Launch threads to process each rank in parallel
    for (int i = 0; i < python_nranks; i++) {
        threads.emplace_back([&input, i]() {
            input[i].weight = modelWeightToGPU(input[i].weight, i);
        });
    }

    // Join all threads to ensure they finish before moving on
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    init(python_nranks, python_vnranks, input, output, pipeTy);

    TensorManager::getInstance().init(python_nranks, python_vnranks);
}

void pllm_update(std::vector<vortexUpdateData>& updateData){
    // std::vector<vortexUpdateData> gpu_update_datas = TensorManager::getInstance().update_data_to_gpu(updateData);

    update(python_nranks, updateData);
}

void pllm_config(std::vector<vortexConfigData>& configData){
    config(python_nranks, configData);
}

void printInputTmpBuffer(vortexInitData &inputData){
    half* tmp_buffer = inputData.tmp_buffer;
    printf("%p\n", tmp_buffer);
    half* host_tmp_buffer = new half[1024];
    cudaMemcpy(host_tmp_buffer, tmp_buffer, 3*sizeof(half), cudaMemcpyDeviceToHost);
    for(int i=0; i<3; i++){
        std::cout << float(host_tmp_buffer[i]) << std::endl;
    }
}


void setGemvBatchSizeUpdate(vortexUpdateData &updateData, pybind11::array_t<int> gemv_batch_size){
    pybind11::buffer_info gemv_batch_size_info = gemv_batch_size.request();
    int *gemv_batch_size_ptr = static_cast<int *>(gemv_batch_size_info.ptr);
    int * new_data = new int[gemv_batch_size_info.shape[0]];
    for (int i = 0; i < gemv_batch_size_info.shape[0]; i++){
        new_data[i] = gemv_batch_size_ptr[i];
    }
    updateData.gemv_batch_size = new_data;
}

void setGemvNumBlocksUpdate(vortexUpdateData &updateData, pybind11::array_t<int> gemv_num_blocks){
    pybind11::buffer_info gemv_num_blocks_info = gemv_num_blocks.request();
    int *gemv_num_blocks_ptr = static_cast<int *>(gemv_num_blocks_info.ptr);
    int * new_data = new int[gemv_num_blocks_info.shape[0]];
    for (int i = 0; i < gemv_num_blocks_info.shape[0]; i++){
        new_data[i] = gemv_num_blocks_ptr[i];
    }
    updateData.gemv_num_blocks = new_data;
}

void setGemmOpTag(vortexConfigData &configData, pybind11::list gemm_op_tag){
    std::vector<std::string> new_data;
    for (size_t i = 0; i < gemm_op_tag.size(); i++){
        new_data.push_back(gemm_op_tag[i].cast<std::string>());
    }
    configData.gemm_op_tag = new_data;
}

void setPtr(vortexWeight &weight, size_t ptr){
    // weight.ptr = new half[weight.N * weight.K];
    // for (int i = 0; i < weight.N * weight.K; i++){
    //     weight.ptr[i] = reinterpret_cast<half*>(ptr)[i];
    // }
    weight.ptr = reinterpret_cast<half*>(ptr);
}

void setModelConfig(const std::string& filename) {
    updateVortexModelConfig(filename);
}

pybind11::array getPipelineOutput(){
    // int num = outputs.size();
    // pybind11::tuple output_tuple(num);
    // for (int i = 0; i < num; i++){
    //     pybind11::array sampled_token_array = pybind11::array(
    //         pybind11::dtype("int32"),
    //         {aggregated_output.size()}, // fix me
    //         {sizeof(int32_t)},
    //         aggregated_output.data()
    //     );

    //     pybind11::array offload_kv_cache = pybind11::array(
    //         pybind11::dtype("f2"),
    //         {16*128*128*80*2}, // fix me
    //         {sizeof(half)},
    //         outputs[i].offload_kv_cache
    //     );

    //     output_tuple[i] = pybind11::make_tuple(sampled_token_array, offload_kv_cache);
    // }
    // 
    pybind11::array sampled_token_array = pybind11::array(
            pybind11::dtype("int32"),
            {aggregated_output.size()}, 
            {sizeof(int32_t)},
            aggregated_output.data()
        );
    return sampled_token_array;
}


PYBIND11_MODULE(pllm_python, m) {

    pybind11::class_<vortexInitData>(m, "VortexInitData")
        .def(pybind11::init<>())
        .def("setKVdata", &setKVdata)
        .def("setWeight", &setWeight)
        .def("setTmpBuffer", &setTmpBuffer)
        .def_readwrite("weight_size", &vortexInitData::weight_size)
        .def_readwrite("tmp_buffer_size", &vortexInitData::tmp_buffer_size)
        ;

    pybind11::class_<vortexOutputData>(m, "VortexOutputData")
        .def(pybind11::init<>());

    pybind11::class_<vortexUpdateData>(m, "VortexUpdateData")
        .def(pybind11::init<>())
        .def_readwrite("decodePrefillBorder", &vortexUpdateData::decodePrefillBorder)
        .def_readwrite("prefillNum", &vortexUpdateData::prefillNum)
        .def_readwrite("prefillTokensNum", &vortexUpdateData::prefillTokensNum)
        .def_readwrite("keepTokenListLength", &vortexUpdateData::keepTokenListLength)
        .def("setKVIndices", &setKVIndicesUpdate)
        .def("setKVIndptr", &setKVIndptrUpdate)
        .def("setKVLastPageLen", &setKVLastPageLenUpdate)
        .def("setInputIndptr", &setInputIndptrUpdate)
        .def("setRevInputIndptr", &setRevInputIndptrUpdate)
        .def("setInputEmbedding", &setInputEmbeddingUpdate)
        .def("setGemvBatchSize", &setGemvBatchSizeUpdate)
        .def("setGemvNumBlocks", &setGemvNumBlocksUpdate)
        .def("setPerTokenOffset", &setPerTokenOffsetUpdate)
        .def("setKeepTokenList", &setKeepTokenListUpdate)
        ;
    pybind11::class_<vortexConfigData>(m, "VortexConfigData")
        .def(pybind11::init<>())
        .def_readwrite("gemmOpTag", &vortexConfigData::gemm_op_tag)
        .def_readwrite("globalBatchSize", &vortexConfigData::global_batch_size)
        .def_readwrite("nanobatch1Size", &vortexConfigData::nanobatch_1_size)
        .def_readwrite("kqv1Size", &vortexConfigData::kqv1_size)
        .def_readwrite("kqv3Size", &vortexConfigData::kqv3_size)
        .def("setGemmOpTag", &setGemmOpTag)
        ;

    pybind11::enum_<Worker::PipelineType>(m, "PipelineType")
        .value("PLLM", Worker::PipelineType::PLLM)
        .value("NONOVERLAP", Worker::PipelineType::NONOVERLAP)
        .value("NANOBATCH", Worker::PipelineType::NANOBATCH)
        .value("PLLMOFFLOAD", Worker::PipelineType::PLLMOFFLOAD)        
        .value("LOCAL", Worker::PipelineType::LOCAL)
        .value("NON_OVERLAP_LOCAL", Worker::PipelineType::NON_OVERLAP_LOCAL)
        .value("NANOBATCH_LOCAL", Worker::PipelineType::NANOBATCH_LOCAL)
        .value("NONOVERLAP_KQVBIAS", Worker::PipelineType::NONOVERLAP_KQVBIAS)
        .value("NANOBATCH_KQVBIAS", Worker::PipelineType::NANOBATCH_KQVBIAS)
        .value("KQVBIAS", Worker::PipelineType::KQVBIAS)
        ;

    pybind11::class_<vortexWeight>(m, "VortexWeight")
        .def(pybind11::init<>())
        .def("setPtr", &setPtr)
        .def_readwrite("N", &vortexWeight::N)
        .def_readwrite("K", &vortexWeight::K)
        ;
    
    pybind11::class_<vortexLayerWeight>(m, "VortexLayerWeight")
        .def(pybind11::init<>())
        .def_readwrite("W_O1", &vortexLayerWeight::W_O1)
        .def_readwrite("W_O2", &vortexLayerWeight::W_O2)
        .def_readwrite("W_U", &vortexLayerWeight::W_U)
        .def_readwrite("W_G", &vortexLayerWeight::W_G)
        .def_readwrite("W_D", &vortexLayerWeight::W_D)
        .def_readwrite("W_KQV", &vortexLayerWeight::W_KQV)
        .def_readwrite("B_KQV", &vortexLayerWeight::B_KQV)
        .def_readwrite("W_LN_Attention", &vortexLayerWeight::W_LN_Attention)
        .def_readwrite("W_LN_FFN", &vortexLayerWeight::W_LN_FFN)
        .def_readwrite("W_ROT", &vortexLayerWeight::W_ROT)
        ;
    
    pybind11::class_<vortexModelWeight>(m, "VortexModelWeight")
        .def(pybind11::init<>())
        .def_readwrite("lm_head", &vortexModelWeight::lm_head)
        .def_readwrite("embedding", &vortexModelWeight::embedding)
        .def_readwrite("model_layernorm", &vortexModelWeight::model_layernorm)
        .def_readwrite("layer_weight", &vortexModelWeight::layer_weight)
        ;

    m.def("init", &pllm_init, "Initializes the Pipeline.");
    m.def("update", &pllm_update, "Updates the Pipeline.");
    m.def("config", &pllm_config, "Configures the Pipeline.");
    m.def("run", &run, "Runs the Pipeline.");
    m.def("finalize", &finalize, "Finalizes the Pipeline.");
    m.def("run_async", &run_async, "Runs the Pipeline asynchronously.");
    m.def("run_async_wait", &run_async_wait, "Waits for the Pipeline to finish.");
    m.def("getPipelineOutput", &getPipelineOutput, "Returns the output of the Pipeline.");
    m.def("setRank", &setRank, "Sets the rank of the Pipeline.");
    m.def("createModelWeight", &createModelWeightCPU, "Creates a weight.");
    m.def("setModelConfig", &setModelConfig, "Sets the model configuration.");
    m.def("createInitData", &createInitData, "Creates an initialization data.");
}