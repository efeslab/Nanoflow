#pragma once
#include "config.h"
#include "eventManager.cuh"
#include "gemmShape.cuh"
#include "gemvWrapper.cuh"
#include "netWrapper.cuh"
#include "networkManager.cuh"
#include "otherWrapper.cuh"
#include "dualWrapper.cuh"
#include "vortexData.cuh"
#include "gemvConfig.cuh"

#include "allocManager.cuh"
#include "gemvDependency.cuh"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"

#define SET_NAME_PTR(var) (var)->name = #var;
#define SET_NAME_REF(var) (var).name = #var;


class PipelineBase {
public:
	PipelineBase(vortexInitData* input_data, int nrank, int nranks, int vnranks);

protected:
	vortexInitData* input_data;
	vortexUpdateData update_data;
	vortexConfigData config_data;

	// static constexpr int numGPU = GPU_NUM;
	int rank;
	int nranks;
	// virtualized number of ranks, should >= nranks. Used to simulate a large setup using fewer number of gpus
	int vnranks;
	std::shared_ptr<mscclpp::Communicator> comm;
	std::vector<std::shared_ptr<mscclpp::Connection>> connections;

	AllocationManager<cutlass::half_t> tmpBufferM;
	EventManager ev;
	std::vector<cudaEvent_t>& events = ev.events;
	std::shared_ptr<spdlog::logger> private_logger;


	void NetOpPrepare();
public:
	virtual vortexOutputData run() = 0;
	virtual void update(vortexUpdateData* update_data) = 0;
	virtual void config(vortexConfigData* config_data) = 0;
	virtual void setName(){}
};

class Pipeline : public PipelineBase {
public:
	enum class GEMM_NAME {
		O1=0,
		O2,
		D1,
		D2,
		KQV1,
		KQV2,
		KQV3,
		KQV4,
		// LOGITS1,
		// LOGITS2,
		LOGITS,
		NUM
	};
	static constexpr std::array gemmConfig = {
		"128_128_32_64_64_32_3_5_RowMajor_RowMajor_RowMajor", // O1
		"128_128_32_64_64_32_1_4_RowMajor_RowMajor_RowMajor", // O2
		"128_128_32_64_64_32_1_5_RowMajor_RowMajor_RowMajor",    // D1
		"128_128_32_64_64_32_1_4_RowMajor_RowMajor_RowMajor",    // D2
		"128_64_64_64_32_64_2_3_RowMajor_RowMajor_RowMajor",        // KQV1
		"128_128_32_64_64_32_2_5_RowMajor_RowMajor_RowMajor",       // KQV2
		"128_128_32_64_64_32_1_5_RowMajor_RowMajor_RowMajor",       // KQV3
		"128_64_64_64_32_64_2_3_RowMajor_RowMajor_RowMajor",       // KQV4
		"128_256_32_64_64_32_2_3_RowMajor_RowMajor_RowMajor"		// LOGITS
	};
	std::array<BaseGEMMWrapper*, static_cast<size_t>(GEMM_NAME::NUM)> gemms;
#define GEMM_ALIAS(name) BaseGEMMWrapper *&name = gemms[static_cast<size_t>(GEMM_NAME::name)]
	GEMM_ALIAS(O1);
	GEMM_ALIAS(O2);
	GEMM_ALIAS(D1);
	GEMM_ALIAS(D2);
	GEMM_ALIAS(KQV1);
	GEMM_ALIAS(KQV2);
	GEMM_ALIAS(KQV3);
	GEMM_ALIAS(KQV4);
	GEMM_ALIAS(LOGITS);

#ifdef ENABLE_NETWORK
	NetAllGather AG_O1;
	NetAllReduce AR_O2;
	NetAllReduce AR_D1;
	NetAllReduceWithLN AR1_D2, AR2_D2;
	NetAllGather AG1_GEMV;
#else
	SleepWrapper AG_O1, AG_O2, RS_D1, AG_D1, AR1_D2, AR2_D2, AG1_GEMV, AG2_GEMV;
#endif

	DualWrapper<128, 64, 32, 64, 32, 32, 1, 5, cutlass::layout::RowMajor, cutlass::layout::RowMajor, cutlass::layout::RowMajor> dual1, dual2;
	GenEmbedding genEmbedding1, genEmbedding2_1, genEmbedding2_2;
	GenEmbedding genEmbedding2_1_partial, genEmbedding2_2_partial;

	DecodeGemvWrapper GEMV1, GEMV2, GEMV3, GEMV4;

	std::array<DecodeGemvWrapper*, 4> gemvs {&GEMV1, &GEMV2, &GEMV3, &GEMV4};

	PrefillGemvWrapper prefill;

	SleepWrapper s;
	cudaStream_t stream_gemm, stream_gemv, stream_net, stream_other, stream_cpy;//cpy: offload to host

	LayerNorm layerNormAttention1, layerNormAttention2_1, layerNormAttention2_2;
	LayerNorm layerNormFFN1, layerNormFFN2;
	LayerNorm layerNormModel1;
	BaseGEMMWrapper* KQV_ptrs[4];


	RoPEAppend roPEAppends[4];

	PageAggregator pageAgg;
	PageDispatcher pageDisp;

	SplitTensor splitTensor;

	MaxSampling maxSampler;

	gemvDependency gemv_dep;

	KeepToken keepToken;
	CopyTensor copyTensor;

	int batch_start = 0;

	half* offloadKVCache;
	int* outputTokens;
	int32_t* finished_idx;
	int32_t* load_idx;
	half* deviceOffloadKVCache;
	half* deviceLoadKVCache;
	
	int update_token_num;
	pllmTensor<half> gemvQ, gemvAggregateOutput, KQV_output;
	vortexOutputData output_data;

	std::vector<pllmTensor<cutlass::half_t>> KQV_biases;

public:
	Pipeline(vortexInitData* input_data, int nrank, int nranks, int vnranks, bool enable_offload = false, bool nanobatch_only = false, bool kqv_bias = false);
	~Pipeline();
	void ScheduleInit();
	void NetOpInit();
	void OtherOpInit();
	void GEMMOpInit();
	void GEMVOpInit();
	void StreamInit();
	void setWeight(int layer);
	void profileGEMM();
	vortexOutputData run() override;
	void update(vortexUpdateData* update_data) override;
	void config(vortexConfigData* config_data) override;

	void GEMVOpUpdate();
	void setName() override;
	bool enable_offload = false;
	bool nanobatch_only = false;
	bool kqv_bias = false;

	void assignSameStream();

private:
	void init();
	double totalCompute();
};

class NonOverlapPipeline : public PipelineBase {
public:
	enum class GEMM_NAME
	{
		O = 0,
		// UG,
		D,
		KQV,
		LOGITS,
		NUM
	};
	static constexpr std::array gemmConfig = {
		"128_128_32_64_64_32_1_5_RowMajor_RowMajor_RowMajor", // O
		"128_128_32_64_64_32_1_5_RowMajor_RowMajor_RowMajor",    // D
		"128_128_32_64_64_32_1_5_RowMajor_RowMajor_RowMajor",       // KQV
		"128_256_32_64_64_32_1_3_RowMajor_RowMajor_RowMajor",    // LOGITS
	};

	static constexpr int gemmNum = static_cast<int>(GEMM_NAME::NUM);
	std::array<BaseGEMMWrapper*, gemmNum> gemms;
	GEMM_ALIAS(O);
	// GEMM_ALIAS(UG);
	GEMM_ALIAS(D);
	GEMM_ALIAS(KQV);
	GEMM_ALIAS(LOGITS);
#ifdef ENABLE_NETWORK
	NetAllGather AG_O;
	NetAllReduce AR_D;
	NetAllGather AG_GEMV;
#else
// TODO
#endif
	DualWrapper<128, 64, 32, 64, 32, 32, 1, 5, cutlass::layout::RowMajor, cutlass::layout::RowMajor, cutlass::layout::RowMajor> dual;
	LayerNorm layerNormAttention;
	LayerNorm layerNormFFN;
	LayerNorm layerNormModel;
	RoPEAppend roPEAppend;
	PrefillGemvWrapper prefill;
	DecodeGemvWrapper GEMV;
	GenEmbedding embedding;
	SplitTensor splitTensor;

	pllmTensor<cutlass::half_t> gemvInput, gemvOutput;
	MaxSampling maxSampler;
	KeepToken keepToken;


	cudaStream_t stream_all;
	int* outputTokens;
	bool kqv_bias = false;
	std::vector<pllmTensor<cutlass::half_t>> KQV_biases;

public:
	NonOverlapPipeline(vortexInitData* input_data, int nrank, int nranks, int vnranks, bool kqv_bias = false);
	void ScheduleInit();
	void GEMMOpInit();
	void GEMVOpInit();
	void OtherOpInit();
	void setWeight(int layer);
	vortexOutputData run() override;
	// update_data is expected to contain only one meaningful batch configuration
	// and the decodePrefillBorder should == that batch size
	void update(vortexUpdateData* update_data) override;
	void config(vortexConfigData* config_data) override;

	void GEMVOpUpdate();
	void setName() override;
	vortexOutputData output_data;
	half* weight_buffer;

private:
	void init();
	double totalCompute();
};




class NonOverlapNanoBatchPipeline : public PipelineBase {
public:
	enum class GEMM_NAME
	{
		O = 0,
		UG,
		D,
		KQV,
		KQV_START,
		NUM
	};
	static constexpr std::array gemmConfig = {
		"128_128_32_64_64_32_1_5_ColumnMajor_RowMajor_ColumnMajor", // O
		"128_128_32_64_64_32_1_5_ColumnMajor_RowMajor_RowMajor",    // UG
		"128_128_32_64_64_32_1_5_ColumnMajor_RowMajor_RowMajor",    // D
		"128_128_32_64_64_32_1_5_RowMajor_RowMajor_RowMajor",       // KQV
		"128_128_32_64_64_32_1_5_ColumnMajor_RowMajor_RowMajor"     // KQV_START
	};

	static constexpr int gemmNum = static_cast<int>(GEMM_NAME::NUM);
	std::array<BaseGEMMWrapper*, gemmNum> gemms;
	GEMM_ALIAS(O);
	GEMM_ALIAS(UG);
	GEMM_ALIAS(D);
	GEMM_ALIAS(KQV);
	GEMM_ALIAS(KQV_START);
#ifdef ENABLE_NETWORK
	NetAllGather AG_O;
	NetAllReduce AR_D;
	NetAllGather AG_GEMV;
#else
// TODO
#endif
	LayerNorm layerNormAttention;
	LayerNorm layerNormFFN;
	Activation activation;
	RoPEAppend roPEAppend;
	PrefillGemvWrapper prefill;
	DecodeGemvWrapper GEMV;
	Transpose O_TR;

	pllmTensor<cutlass::half_t> gemvInput, gemvOutput;

	cudaStream_t stream_all;
	int* outputTokens;

public:
	NonOverlapNanoBatchPipeline(vortexInitData* input_data, int nrank, int nranks, int vnranks);
	void ScheduleInit();
	void GEMMOpInit();
	void GEMVOpInit() { /* Nop */ }
	void OtherOpInit() { /* Nop */ }
	void setWeight(int layer);
	vortexOutputData run() override;
	// update_data is expected to contain only one meaningful batch configuration
	// and the decodePrefillBorder should == that batch size
	void update(vortexUpdateData* update_data) override;
	void config(vortexConfigData* config_data) override;

	void GEMVOpUpdate();

private:
	void init();
	double totalCompute();
};

class LocalPipeline : public PipelineBase {
public:
	void assignSameStream();
	enum class GEMM_NAME
	{
		O1=0,
		O2,
		D1,
		D2,
		KQV1,
		KQV2,
		LOGITS,
		NUM
	};
	static constexpr std::array gemmConfig = {
		"128_256_64_64_64_64_1_3_ColumnMajor_ColumnMajor_RowMajor", // O
		"128_256_64_64_64_64_1_3_ColumnMajor_ColumnMajor_RowMajor", // O
		"128_256_32_64_64_32_1_3_ColumnMajor_ColumnMajor_RowMajor",    // D
		"128_256_32_64_64_32_1_3_ColumnMajor_ColumnMajor_RowMajor",    // D
		"128_128_32_64_64_32_1_5_ColumnMajor_ColumnMajor_RowMajor",       // KQV
		"128_128_32_64_64_32_1_5_ColumnMajor_ColumnMajor_RowMajor",       // KQV
		"128_256_32_64_64_32_1_3_ColumnMajor_ColumnMajor_RowMajor",    // LOGITS
	};

	static constexpr int gemmNum = static_cast<int>(GEMM_NAME::NUM);
	std::array<BaseGEMMWrapper*, gemmNum> gemms;
	GEMM_ALIAS(O1);
	GEMM_ALIAS(O2);
	GEMM_ALIAS(D1);
	GEMM_ALIAS(D2);
	GEMM_ALIAS(KQV1);
	GEMM_ALIAS(KQV2);
	GEMM_ALIAS(LOGITS);

	DualWrapper<128, 64, 32, 64, 32, 32, 1, 5, cutlass::layout::RowMajor, cutlass::layout::RowMajor, cutlass::layout::RowMajor> dual1, dual2;
	GenEmbedding genEmbedding1, genEmbedding2;
	DecodeGemvWrapper GEMV1, GEMV2, GEMV3, GEMV4;
	std::array<DecodeGemvWrapper*, 4> gemvs {&GEMV1, &GEMV2, &GEMV3, &GEMV4};
	LayerNorm layerNormAttention1, layerNormAttention2;
	LayerNorm layerNormFFN1, layerNormFFN2;
	LayerNorm layerNormModel;
	RoPEAppend roPEAppends[2]; 
	PrefillGemvWrapper prefill1, prefill2;
	MaxSampling maxSampler;
	KeepToken keepToken;

	pllmTensor<half> gemvQ1, gemvQ2;
	pllmTensor<half> gemvAggregateOutput1, gemvAggregateOutput2;
	pllmTensor<half> KQV_output1, KQV_output2;
	BaseGEMMWrapper* KQV_ptrs[2]; // KQV1, KQV2
	vortexOutputData output_data;
	cudaStream_t stream_gemm;
	cudaStream_t stream_gemv;
	cudaStream_t stream_other;
	int update_token_num;
	vortexUpdateData prefill1_update;
	bool nanobatch_only = false;

public:
	LocalPipeline(vortexInitData* input_data, int nrank, int nranks, int vnranks, bool nanobatch_only = false);
	void ScheduleInit();
	void GEMMOpInit();
	void GEMVOpInit();
	void OtherOpInit();
	void setWeight(int layer);
	void setWeight_group_O(int layer);
	void setWeight_group_LN(int layer);
	vortexOutputData run() override;
	// update_data is expected to contain only one meaningful batch configuration
	// and the decodePrefillBorder should == that batch size
	void update(vortexUpdateData* update_data) override;
	void config(vortexConfigData* config_data) override;

	void GEMVOpUpdate();
	void setName() override;
	

private:
	void init();
	double totalCompute();
	int* host_rev_input_indptr;
};

class NonOverlapLocalPipeline : public PipelineBase {
public:
	enum class GEMM_NAME
	{
		O=0,
		D,
		KQV,
		LOGITS,
		NUM
	};
	static constexpr std::array gemmConfig = {
		"128_256_64_64_64_64_1_3_ColumnMajor_ColumnMajor_RowMajor", // O
		"128_256_32_64_64_32_1_3_ColumnMajor_ColumnMajor_RowMajor",    // D
		"128_128_32_64_64_32_1_5_ColumnMajor_ColumnMajor_RowMajor",       // KQV
		"128_256_32_64_64_32_1_3_ColumnMajor_ColumnMajor_RowMajor",    // LOGITS
	};

	static constexpr int gemmNum = static_cast<int>(GEMM_NAME::NUM);
	std::array<BaseGEMMWrapper*, gemmNum> gemms;
	GEMM_ALIAS(O);
	GEMM_ALIAS(D);
	GEMM_ALIAS(KQV);
	GEMM_ALIAS(LOGITS);

	DualWrapper<128, 64, 32, 64, 32, 32, 1, 5, cutlass::layout::RowMajor, cutlass::layout::RowMajor, cutlass::layout::RowMajor> dual;
	GenEmbedding genEmbedding;
	DecodeGemvWrapper GEMV;
	LayerNorm layerNormAttention;
	LayerNorm layerNormFFN;
	LayerNorm layerNormModel;
	RoPEAppend roPEAppend; 
	PrefillGemvWrapper prefill;
	MaxSampling maxSampler;
	KeepToken keepToken;

	pllmTensor<cutlass::half_t> gemvInput;
	pllmTensor<cutlass::half_t> gemvOutput;
	pllmTensor<half> KQV_output;
	vortexOutputData output_data;
	cudaStream_t stream_all;
	int* outputTokens;
	int update_token_num;

public:
	NonOverlapLocalPipeline(vortexInitData* input_data, int nrank, int nranks, int vnranks);
	void ScheduleInit();
	void GEMMOpInit();
	void GEMVOpInit();
	void OtherOpInit();
	void setWeight(int layer);
	vortexOutputData run() override;
	// update_data is expected to contain only one meaningful batch configuration
	// and the decodePrefillBorder should == that batch size
	void update(vortexUpdateData* update_data) override;
	void config(vortexConfigData* config_data) override;

	void GEMVOpUpdate();
	void setName() override;

private:
	void init();
	double totalCompute();
	int* host_rev_input_indptr;
};