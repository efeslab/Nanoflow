#pragma once

#include "cutlass/cutlass.h"
#include "cutlassGemmBase.cuh"

// #if __CUDA_ARCH__ >= 900

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"

using ColumnMajor = cutlass::layout::ColumnMajor;
using RowMajor = cutlass::layout::RowMajor;
template <int cta_m,
		  int cta_n,
		  int cta_k,
		  int cluster_m,
		  int cluster_n,
		  int cluster_k,
		  int split_k,
		  typename LayoutInputA_ = cutlass::layout::ColumnMajor,
		  typename LayoutInputB_ = cutlass::layout::ColumnMajor,
		  typename LayoutOutput_ = cutlass::layout::ColumnMajor,
		  typename EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto,
		  typename MainloopSchedule = cutlass::gemm::collective::KernelScheduleAuto,
		  typename GEMMSchedule = cutlass::gemm::PersistentScheduler
		  >
struct CutlassH100GEMMWrapper : public BaseGEMMWrapperTemplate<LayoutInputA_, LayoutInputB_, LayoutOutput_> {
	using BaseGEMMWrapper = BaseGEMMWrapperTemplate<LayoutInputA_, LayoutInputB_, LayoutOutput_>::BaseGEMMWrapper;
    using BaseGEMMWrapper::M;
    using BaseGEMMWrapper::N;
    using BaseGEMMWrapper::K;
	using BaseGEMMWrapper::name;
	using BaseGEMMWrapper::stream;
    using typename BaseGEMMWrapper::ElementInputA;
    using typename BaseGEMMWrapper::ElementInputB;
    using typename BaseGEMMWrapper::ElementOutput;
	using typename BaseGEMMWrapper::ElementAccumulator;
    using BaseGEMMWrapperT= BaseGEMMWrapperTemplate<LayoutInputA_, LayoutInputB_, LayoutOutput_>;
    using BaseGEMMWrapperT::input_a;
    using BaseGEMMWrapperT::input_b;
    using BaseGEMMWrapperT::input_c;
    using BaseGEMMWrapperT::output_d;
    using BaseGEMMWrapperT::tensor_a_ref;
    using BaseGEMMWrapperT::tensor_b_ref;
    using BaseGEMMWrapperT::tensor_c_ref;
    using BaseGEMMWrapperT::tensor_d_ref;
    using BaseGEMMWrapperT::kLda;
    using BaseGEMMWrapperT::kLdb;
    using BaseGEMMWrapperT::kLdc;
    using BaseGEMMWrapperT::kLdd;

	using ElementComputeEpilogue = ElementAccumulator;

	using LayoutInputA = LayoutInputA_;
	using LayoutInputB = LayoutInputB_;
	using LayoutOutput = LayoutOutput_;

	using MMAOp = cutlass::arch::OpClassTensorOp;

	cutlass::gemm::GemmCoord problem_size;

	using ShapeMMAThreadBlock = cute::Shape<cute::Int<cta_m>, cute::Int<cta_n>, cute::Int<cta_k>>;
	using ShapeCluster = cute::Shape<cute::Int<cluster_m>, cute::Int<cluster_n>, cute::Int<cluster_k>>;
	using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;

	using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

	using cutlass3x_epilogue =
	typename cutlass::epilogue::collective::CollectiveBuilder<
		cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
		ShapeMMAThreadBlock,
		ShapeCluster,
		cutlass::epilogue::collective::EpilogueTileAuto,
		float, float,
		ElementOutput, LayoutOutput, 8,
		ElementOutput, LayoutOutput, 8,
		EpilogueSchedule,
		
		cutlass::epilogue::fusion::LinearCombination<
			ElementOutput,
			float,
			ElementOutput,
			float
		>
	>::CollectiveOp;

	using cutlass3x_main_loop =
	typename cutlass::gemm::collective::CollectiveBuilder<
		cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
		ElementInputA, LayoutInputA, 8,
		ElementInputB, LayoutInputB, 8,
		ElementAccumulator,
		ShapeMMAThreadBlock,
		ShapeCluster,
		cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename cutlass3x_epilogue::SharedStorage))>,
		MainloopSchedule
	>::CollectiveOp;

	using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
		cute::Shape<int,int,int,int>,
		cutlass3x_main_loop,
		cutlass3x_epilogue,
		GEMMSchedule>;

	constexpr static bool isSplit = split_k > 1;
	using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>; 
	using StrideA = typename Gemm::GemmKernel::StrideA;
	using StrideB = typename Gemm::GemmKernel::StrideB;
	using StrideC = typename Gemm::GemmKernel::StrideC;
	using StrideD = typename Gemm::GemmKernel::StrideD;
	StrideA stride_A;
	StrideB stride_B;
	StrideC stride_C;
	StrideD stride_D;
	// uint8_t* workspace;
	cutlass::device_memory::allocation<uint8_t> workspace;

	ElementComputeEpilogue alpha, beta;
	Gemm gemm_op;
	bool inited = false;

	CutlassH100GEMMWrapper()
		: problem_size(M, N, K)
		, alpha(1)
		, beta(0) { }

	void work() {
		if (M !=0) {
			// gemm_op(stream);
			gemm_op();
		}
	}


	inline bool isInitialized() const {
		return (problem_size.m() != 0);
	}
	virtual ~CutlassH100GEMMWrapper() override {
		// If workspace was allocated, release it.
		if (inited) {
			workspace.reset();  // This releases the allocated device memory.
		}
	}

	void updateArgument(){
		problem_size = cutlass::gemm::GemmCoord({M, N, K});
		typename Gemm::Arguments  arguments{
			cutlass::gemm::GemmUniversalMode::kGemm,
			{M, N, K, split_k},
			{tensor_a_ref.data(), stride_A, tensor_b_ref.data(), stride_B},
			{{alpha, beta}, tensor_c_ref.data(), stride_C, tensor_d_ref.data(), stride_D},
		};
		cutlass::Status status;
		// status = gemm_op.update(arguments, workspace.get());
		status = gemm_op.initialize(arguments, workspace.get());
		CUTLASS_CHECK(status);
	}
	
	void init_for_stream_k(){
		typename Gemm::Arguments  arguments{
			cutlass::gemm::GemmUniversalMode::kGemm,
			{M, N, K, split_k},
			{tensor_a_ref.data(), stride_A, tensor_b_ref.data(), stride_B},
			{{alpha, beta}, tensor_c_ref.data(), stride_C, tensor_d_ref.data(), stride_D},
		};
		cutlass::Status status;
		status = gemm_op.initialize(arguments, workspace.get());
		CUTLASS_CHECK(status);
	}
	// Expected usage:
	// Call set{A,B,C,D} to configure the input/output tensors before calling this init.
	// Assuming all tensor operands are setup.
	void init(ElementComputeEpilogue beta_) override {
		int M = int(this->M);
		int N = int(this->N);
		int K = int(this->K);
		problem_size = cutlass::gemm::GemmCoord({M,N,K});
		// spdlog::info("name:{} M:{}, N:{}, K:{}, a, b, c, d: {}, {}, {}, {}", name, M, N, K, (size_t)tensor_a_ref.data(),  (size_t)tensor_b_ref.data(),  (size_t)tensor_c_ref.data(),  (size_t)tensor_d_ref.data());
		// spdlog::info("lda, ldb, ldc, ldd: {}, {}, {}, {}", kLda, kLdb, kLdc, kLdd);
		// FIX: only row-row-row is correct
		if (std::is_same_v<LayoutInputA, cutlass::layout::RowMajor>) {
			stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
		}
		else {
			stride_A = cutlass::make_cute_packed_stride(StrideA{}, {K, M, 1});
		}
		if (std::is_same_v<LayoutInputB, cutlass::layout::RowMajor>) {
			stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
		} else {
			stride_B = cutlass::make_cute_packed_stride(StrideB{}, {K, N, 1});
		}
		if (std::is_same_v<LayoutOutput, cutlass::layout::RowMajor>) {
			stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
			stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
		} else {
			stride_C = cutlass::make_cute_packed_stride(StrideC{}, {N, M, 1});
			stride_D = cutlass::make_cute_packed_stride(StrideD{}, {N, M, 1});
		} 


		beta = beta_;
		typename Gemm::Arguments  arguments{
			cutlass::gemm::GemmUniversalMode::kGemm,
			{M, N, K, split_k},
			{tensor_a_ref.data(), stride_A, tensor_b_ref.data(), stride_B},
			{{alpha, beta}, tensor_c_ref.data(), stride_C, tensor_d_ref.data(), stride_D},
		};
		size_t workspace_size = Gemm::get_workspace_size(arguments);
		// cudaMalloc(&workspace, workspace_size);
		workspace = cutlass::device_memory::allocation<uint8_t>(workspace_size);
		// spdlog::info("workspace size: {}", workspace_size);
		cutlass::Status status = gemm_op.can_implement(arguments);
		CUTLASS_CHECK(status);
		// status = gemm_op.initialize(arguments, workspace);
		  CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));

		CUTLASS_CHECK(status);
		inited = true;
	}

	void set_weight(ElementInputB* data_b) {

		tensor_b_ref = cutlass::TensorRef<ElementInputB, LayoutInputB>(data_b, kLdb);
		// Only update the gemm_op if we have already initialized this GEMM
		if(!inited) return; // kan: isInitilized not work if nrank != vnrank. use this inited can temporarily solve this problem.
		updateArgument();
		input_b = data_b;
	}

	void set_alpha(float alpha_) override {
		alpha = alpha_;
		if(!inited) return; // kan: isInitilized not work if nrank != vnrank. use this inited can temporarily solve this problem.
		updateArgument();
	}

	void set_beta(float beta_) {
		beta = beta_;
		if(!inited) return; // kan: isInitilized not work if nrank != vnrank. use this inited can temporarily solve this problem.
		updateArgument();
	}

	bool set_weight(vortexWeight& weight) {
		if(weight.size() != K * N) {
			std::cerr << "Weight size mismatch  " << weight.size() << " " << K << " " << N << std::endl;
			return false;
		}
		set_weight( (ElementInputB*)(weight.ptr) );
		return true;
	}

	bool checkResult() {
		// cutlass::reference::device::Gemm<ElementInputA,
		// 								 LayoutInputA,
		// 								 ElementInputB,
		// 								 LayoutInputB,
		// 								 ElementOutput,
		// 								 LayoutOutput,
		// 								 ElementComputeEpilogue,
		// 								 ElementComputeEpilogue>
		// 	gemm_device;
		// cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d_standard(problem_size.mn());
		// tensor_d_standard.sync_device();
		// gemm_device(problem_size,
		// 			alpha,
		// 			tensor_a_ref,
		// 			tensor_b_ref,
		// 			beta,
		// 			tensor_c_ref,
		// 			tensor_d_standard.device_ref());
		// cudaDeviceSynchronize();
		// tensor_d_standard.sync_host();

		// ElementOutput* data_d = new ElementOutput[M * N];
		// cudaMemcpy(
		// 	data_d, tensor_d_ref.data(), sizeof(ElementOutput) * M * N, cudaMemcpyDeviceToHost);

		// bool passed = true;

		// for(size_t i = 0; i < M * N; i++) {
		// 	if((abs(data_d[i] - tensor_d_standard.host_data()[i]) -0.01) / abs(tensor_d_standard.host_data()[i]) > 1e-1) {
		// 		passed = false;
		// 		spdlog::error("i: {}, d: {}, standard: {}", i,
		// 				static_cast<double>(data_d[i]),
		// 				static_cast<double>(tensor_d_standard.host_data()[i]));
		// 		break;
		// 	}
		// }

		return true;
	}

	float gflops(double runtime_ms) {
		return problem_size.product() * 2.0 / 1e6 / runtime_ms;
	}

	void profile() {
		cudaEvent_t start, end;
		cudaEventCreate(&start);
		cudaEventCreate(&end);

		cudaEventRecord(start);
		constexpr int iter = 1000;
		for(size_t i = 0; i < iter; i++) {
			gemm_op();
		}
		cudaEventRecord(end);
		cudaEventSynchronize(end);

		float runtime_ms = 0;
		cudaEventElapsedTime(&runtime_ms, start, end);
		runtime_ms /= iter;

		std::cout << "shape: " << M << " " << N << " " << K << std::endl;
		std::cout << "cta: " << cta_m << " " << cta_n << " " << cta_k << std::endl;
		std::cout << "cluster: " << cluster_m << " " << cluster_n << " " << cluster_k << std::endl;
		std::cout << "split: " << split_k << std::endl;

		std::cout << "runtime: " << runtime_ms << " ms" << std::endl;
		std::cout << "gflops: " << gflops(runtime_ms) << std::endl;

		cudaEventDestroy(start);
		cudaEventDestroy(end);

		if(!checkResult()) {
			std::cout << "check failed" << std::endl;
		} else {
			std::cout << "check passed" << std::endl;
		}

		cudaError_t error = cudaGetLastError();
		if(error != cudaSuccess) {
			std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
		}

		std::cout << "--------------------------------------" << std::endl;
	}

	double totalCompute() {
		return 2.0 * problem_size.m() * problem_size.n() * problem_size.k();
	}

};


// #else

// template <int cta_m,
// 		  int cta_n,
// 		  int cta_k,
// 		  int cluster_m,
// 		  int cluster_n,
// 		  int cluster_k,
// 		  int warp_m,
// 		  int warp_n,
// 		  int warp_k,
// 		  int split_k,
// 		  typename LayoutInputA_ = cutlass::layout::ColumnMajor,
// 		  typename LayoutInputB_ = cutlass::layout::ColumnMajor,
// 		  typename LayoutOutput_ = cutlass::layout::ColumnMajor>
// struct CutlassH100GEMMWrapper : public BaseGEMMWrapperTemplate<LayoutInputA_, LayoutInputB_, LayoutOutput_> {
// 	void set_alpha(float alpha_) override {
// 	}
// 	void set_beta(float beta_) {
// 	}
// 	bool set_weight(vortexWeight& weight) {
// 	}
// 	void set_weight(ElementInputB* data_b) {
// 	}
// 	void updateArgument(){
// 	}
// 	void work() {
// 		spdlog::error("CUDA ARCH = {} does not support SM90 kernels", __CUDA_ARCH__);
// 	}
// 	void init(ElementAccumulator beta_) override {
// 	}
// 	void profile() {
// 	}
// 	double totalCompute() {
// 	}
// 	float gflops(double runtime_ms) {
// 	}
// 	bool checkResult() {
// 	}
// };

// #endif