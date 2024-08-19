#pragma once
#include "config.h"
#include "eventManager.cuh"
#include "gemmShape.cuh"
#include "gemvWrapper.cuh"
#include "netWrapper.cuh"
#include "networkManager.cuh"
#include "otherWrapper.cuh"
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

	static constexpr int numGPU = GPU_NUM;
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
private:

	std::array<BaseGEMMWrapper*, gemmNum> gemms;
#define GEMM_ALIAS(name) BaseGEMMWrapper *&name = gemms[static_cast<size_t>(GEMM_NAME::name)]
	GEMM_ALIAS(O1);
	GEMM_ALIAS(O2);
	GEMM_ALIAS(UG1);
	GEMM_ALIAS(D1);
	GEMM_ALIAS(UG2);
	GEMM_ALIAS(D2);
	GEMM_ALIAS(KQV1);
	GEMM_ALIAS(KQV2);
	GEMM_ALIAS(KQV3);
	GEMM_ALIAS(KQV4);
	GEMM_ALIAS(LOGITS1);
	GEMM_ALIAS(LOGITS2);

#ifdef ENABLE_NETWORK
	NetAllGather AG_O1;
	NetAllReduce AR_O2;
	NetAllReduce AR_D1;
	NetAllReduceWithLN AR1_D2, AR2_D2;
	NetAllGather AG1_GEMV;
#else
	SleepWrapper AG_O1, AG_O2, RS_D1, AG_D1, AR1_D2, AR2_D2, AG1_GEMV, AG2_GEMV;
#endif

	GenEmbedding genEmbedding1, genEmbedding2_1, genEmbedding2_2;
	GenEmbedding genEmbedding2_1_partial, genEmbedding2_2_partial;

	DecodeGemvWrapper GEMV1, GEMV2, GEMV3, GEMV4;

	std::array<DecodeGemvWrapper*, gemvNum> gemvs {&GEMV1, &GEMV2, &GEMV3, &GEMV4};

	PrefillGemvWrapper prefill; // TODO

	SleepWrapper s;
	cudaStream_t stream_gemm, stream_gemv, stream_net, stream_other, stream_cpy;

	LayerNorm layerNormAttention1, layerNormAttention2_1, layerNormAttention2_2;
	LayerNorm layerNormFFN1, layerNormFFN2;
	LayerNorm layerNormModel1, layerNormModel2_1, layerNormModel2_2;
	BaseGEMMWrapper* KQV_ptrs[4];

	Activation activation1;
	Activation activation2;

	RoPEAppend roPEAppends[4];

	PageAggregator pageAgg;
	PageDispatcher pageDisp;

	SplitTensor splitTensor;

	MaxSampling maxSampler1;
	MaxSampling maxSampler2;

	gemvDependency gemv_dep;

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

public:
	Pipeline(vortexInitData* input_data, int nrank, int nranks, int vnranks, bool enable_offload = false);
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

private:
	void init();
	double totalCompute();
};

class NonOverlapPipeline : public PipelineBase {
private:
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
	NonOverlapPipeline(vortexInitData* input_data, int nrank, int nranks, int vnranks);
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
	// void setName() override;

private:
	void init();
	double totalCompute();
};




class NonOverlapNanoBatchPipeline : public PipelineBase {
private:
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