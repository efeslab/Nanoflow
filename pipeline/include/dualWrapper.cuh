#pragma once
#include <iostream>
#include <span>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#include "cutlass/gemm_coord.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/epilogue/thread/scale_type.h"
#include "cutlass/epilogue/thread/linear_combination_params.h"

#include "device/dual_gemm.h"
#include "kernel/dual_gemm.h"
#include "thread/left_silu_and_mul.h"
#include "threadblock/dual_epilogue.h"
#include "threadblock/dual_mma_base.h"
#include "threadblock/dual_mma_multistage.h"


#include "helper.h"
// #include "config.h"
#include "vortexData.cuh"

#include "operatorWrapper.cuh"

static constexpr bool kUseBias = false;
static constexpr bool kSplitKSerial = false;
static constexpr auto kScaleType = kUseBias ? cutlass::epilogue::thread::ScaleType::NoBetaScaling : (
		// No bias
		kSplitKSerial ? cutlass::epilogue::thread::ScaleType::Default : cutlass::epilogue::thread::ScaleType::Nothing
		);
constexpr bool kStoreD0 = false; 
constexpr bool kStoreD1 = false;

template <int cta_m,
		  int cta_n,
		  int cta_k,
		  int warp_m,
		  int warp_n,
		  int warp_k,
		  int split_k,
		  int stages,
		  typename LayoutInputA_ = cutlass::layout::ColumnMajor,
		  typename LayoutInputB_ = cutlass::layout::ColumnMajor,
		  typename LayoutOutput_ = cutlass::layout::ColumnMajor>
struct DualWrapper : public OperatorWrapper {

	int M, N, K;
	void set_shape(int m, int n, int k) {
		M = m;
		N = n;
		K = k;
		spdlog::info("name:{} M:{}, N:{}, K:{}", name, M, N, K);
	}

	constexpr static int stage_ = stages;
	using LayoutInputA = LayoutInputA_;
	using LayoutInputB = LayoutInputB_;
	using LayoutOutput = LayoutOutput_;

	using MMAOp = cutlass::arch::OpClassTensorOp;

	using SmArch = cutlass::arch::Sm80;

	cutlass::gemm::GemmCoord problem_size;

	using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<cta_m, cta_n, cta_k>;
	using ShapeMMAWarp = cutlass::gemm::GemmShape<warp_m, warp_n, warp_k>;
	using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;

	using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;


	using ElementOperandA = cutlass::half_t;
	using ElementOperandB = cutlass::half_t;
	using ElementOutput = cutlass::half_t;
	using ElementAccumulator = float;
	using ElementCompute = float;




	using EpilogueOutputOp0 = cutlass::epilogue::thread::LinearCombination<
			ElementOutput,
			128 / cutlass::sizeof_bits<ElementOutput>::value,
			ElementAccumulator,
			ElementCompute,
			kScaleType
			>;
	using EpilogueOutputOp1 = cutlass::epilogue::thread::LinearCombination<
			ElementOutput,
			128 / cutlass::sizeof_bits<ElementOutput>::value,
			ElementAccumulator,
			ElementCompute,
			kScaleType
			>;
	using EpilogueOutputOp2 = cutlass::epilogue::thread::LeftSiLUAndMul<
		ElementOutput,
		128 / cutlass::sizeof_bits<ElementOutput>::value,
		ElementOutput,
		ElementCompute
		>;

	const ElementCompute alpha0 = ElementCompute(1);
	const ElementCompute beta0 = ElementCompute(kUseBias ? 1 : 0);
	const ElementCompute alpha1 = ElementCompute(1);
	const ElementCompute beta1 = ElementCompute(kUseBias ? 1 : 0);


	using DualGemm = cutlass::gemm::device::DualGemm<
		ElementOperandA,
		LayoutInputA,
		ElementOperandB,
		LayoutInputB,
		LayoutInputB,
		ElementOutput,
		LayoutOutput,
		ElementAccumulator,
		MMAOp,
		SmArch,
		ShapeMMAThreadBlock,
		ShapeMMAWarp,
		ShapeMMAOp,
		EpilogueOutputOp0,
		EpilogueOutputOp1,
		EpilogueOutputOp2,
		SwizzleThreadBlock,
		stage_,
		kStoreD0,
		kStoreD1,
		kSplitKSerial
	>;
	
	
	using TensorRefA = cutlass::TensorRef<ElementOperandA, LayoutInputA>;
	using TensorRefB = cutlass::TensorRef<ElementOperandB, LayoutInputB>;
	using TensorRefC = cutlass::TensorRef<ElementOutput, LayoutOutput>;
	using TensorRefD = cutlass::TensorRef<ElementOutput, LayoutOutput>;

	TensorRefA tensor_a_ref;
	TensorRefB tensor_b0_ref;
	TensorRefB tensor_b1_ref;
	TensorRefC tensor_c_ref;
	TensorRefD tensor_d0_ref;
	TensorRefD tensor_d1_ref;
	TensorRefD tensor_d2_ref;

	pllmTensor<ElementOperandA> pllm_tensor_a;
	pllmTensor<ElementOperandB> pllm_tensor_b1;
	pllmTensor<ElementOperandB> pllm_tensor_b2;
	pllmTensor<ElementOutput> pllm_tensor_c;
	pllmTensor<ElementOutput> pllm_tensor_d0;
	pllmTensor<ElementOutput> pllm_tensor_d1;
	pllmTensor<ElementOutput> pllm_tensor_d2;


	cutlass::device_memory::allocation<uint8_t> workspace;



	const int& kLda = std::is_same_v<LayoutInputA, cutlass::layout::RowMajor> ? K : M;
	const int& kLdb = std::is_same_v<LayoutInputB, cutlass::layout::RowMajor> ? N : K;
	const int& kLdc = std::is_same_v<LayoutOutput, cutlass::layout::RowMajor> ? N : M;
	const int& kLdd = kLdc;

	DualGemm gemm_op;
	constexpr static size_t smem_size = sizeof(typename DualGemm::SharedStorage);
	bool inited = false;
	DualWrapper()
		: problem_size(M, N, K)
		, tensor_a_ref()
		, tensor_c_ref()
		, tensor_d0_ref()
		, tensor_d1_ref()
		, tensor_d2_ref()
	{

	}

	void work() override {
		gemm_op(stream);
	}

	DualWrapper & setOutput(pllmTensor<ElementOutput> data) {
		return setD(data);
	}

	DualWrapper& setA(pllmTensor<ElementOperandA> data_a) {
		if (std::is_same_v<LayoutInputA, cutlass::layout::RowMajor>) {
			assert(data_a.layout == PllmLayout::ROW_MAJOR);
			assert(data_a.dim1 == M);
			assert(data_a.dim2 == K);
		} else {
			assert(data_a.layout == PllmLayout::COL_MAJOR);
			assert(data_a.dim1 == K);
			assert(data_a.dim2 == M);
		}
		tensor_a_ref = TensorRefA(data_a.ptr, kLda);
		pllm_tensor_a = data_a;
		return *this;
	}
	DualWrapper& setC(pllmTensor<ElementOutput> data_c) {
		if (std::is_same_v<LayoutOutput, cutlass::layout::RowMajor>) {
			assert(data_c.layout == PllmLayout::ROW_MAJOR);
			assert(data_c.dim1 == M);
			assert(data_c.dim2 == N);
		} else {
			assert(data_c.layout == PllmLayout::COL_MAJOR);
			assert(data_c.dim1 == N);
			assert(data_c.dim2 == M);
		}
		tensor_c_ref = TensorRefC(data_c.ptr, kLdc);
		pllm_tensor_c = data_c;
		return *this;
	}
	DualWrapper& setD(pllmTensor<ElementOutput> data_d0, pllmTensor<ElementOutput> data_d1, pllmTensor<ElementOutput> data_d2) {
		tensor_d0_ref = TensorRefD(data_d0.ptr, kLdd);
		tensor_d1_ref = TensorRefD(data_d1.ptr, kLdd);
		tensor_d2_ref = TensorRefD(data_d2.ptr, kLdd);
		pllm_tensor_d0 = data_d0;
		pllm_tensor_d1 = data_d1;
		pllm_tensor_d2 = data_d2;
		
		return *this;
	}
	pllmTensor<ElementOperandA> getA() {
		return pllm_tensor_a;
	}
	pllmTensor<ElementOutput> getD() {
		return pllm_tensor_d2;
	}
	pllmTensor<ElementOutput> getC() {
		return pllm_tensor_c;
	}

	inline bool isInitialized() const {
		return (problem_size.m() != 0);
	}
	typename cutlass::TensorRef<ElementOutput, LayoutOutput> nullptr_ref{};
	void init() {
		problem_size = cutlass::gemm::GemmCoord({M, N, K});
		spdlog::info("problem_size: {}, {}, {}", problem_size.m(), problem_size.n(), problem_size.k());
		typename DualGemm::Arguments arguments{
			cutlass::gemm::DualGemmMode::kGemm,
			problem_size,
			tensor_a_ref,
			tensor_b0_ref,
			tensor_c_ref,
			kStoreD0 ? tensor_d0_ref : nullptr_ref,
			tensor_b1_ref,
			tensor_c_ref,
			kStoreD1 ? tensor_d1_ref : nullptr_ref,
			tensor_d2_ref,
			{alpha0, beta0},
			{alpha1, beta1},
			{},
			split_k};
		size_t workspace_size = DualGemm::get_workspace_size(arguments);
		workspace = cutlass::device_memory::allocation<uint8_t>(workspace_size);
		cutlass::Status status = gemm_op.can_implement(arguments);
		CUTLASS_CHECK(status);
		status = gemm_op.initialize(arguments, workspace.get());
		CUTLASS_CHECK(status);
		inited = true;
	}

	bool set_weight(vortexWeight& weight1, vortexWeight& weight2) {
		tensor_b0_ref = TensorRefB((ElementOperandB*)(weight1.ptr), kLdb);
		tensor_b1_ref = TensorRefB((ElementOperandB*)(weight2.ptr), kLdb);
		if(!inited) return true;
		typename DualGemm::Arguments arguments{
			cutlass::gemm::DualGemmMode::kGemm,
			problem_size,
			tensor_a_ref,
			tensor_b0_ref,
			tensor_c_ref,
			kStoreD0 ? tensor_d0_ref : nullptr_ref,
			tensor_b1_ref,
			tensor_c_ref,
			kStoreD1 ? tensor_d1_ref : nullptr_ref,
			tensor_d2_ref,
			{alpha0, beta0},
			{alpha1, beta1},
			{},
			split_k};
		pllm_tensor_b1 = pllmTensor<ElementOperandB>((ElementOperandB*)(weight1.ptr), K, N, PllmLayout::ROW_MAJOR);
		pllm_tensor_b2 = pllmTensor<ElementOperandB>((ElementOperandB*)(weight2.ptr), K, N, PllmLayout::ROW_MAJOR);
		cutlass::Status status;
		status = gemm_op.update(arguments);
		CUTLASS_CHECK(status);
		return true;
	}

	OperatorWrapper& logImpl(std::shared_ptr<spdlog::logger> logger = default_logger) override{
		log_tensor(logger, name+" input", pllm_tensor_a, 10, 20);
		log_tensor(logger, name+" weight0", pllm_tensor_b1, 10, 20);
		log_tensor(logger, name+" weight0", pllm_tensor_b2, 10, 20);
		if (pllm_tensor_c.dim1 != 0 && pllm_tensor_c.dim2 != 0) {
			log_tensor(logger, name+" C", pllm_tensor_c, 10, 20);
		}
		else{
			logger->info("C is not set");
		}
		log_tensor(logger, name+" output0", pllm_tensor_d0, 10, 20);
		log_tensor(logger, name+" output1", pllm_tensor_d1, 10, 20);
		log_tensor(logger, name+" output2", pllm_tensor_d2, 128, 20);
		return *this;
	}

};