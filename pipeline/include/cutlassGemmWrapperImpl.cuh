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

#include "helper.h"
// #include "config.h"
#include "vortexData.cuh"

#include "operatorWrapper.cuh"

struct BaseGEMMWrapper : public OperatorWrapper{
	using ElementInputA = cutlass::half_t;
	using ElementInputB = cutlass::half_t;
	using ElementOutput = cutlass::half_t;
	using ElementAccumulator = float;

	size_t M, N, K;
	void set_shape(int m, int n, int k) {
		M = m;
		N = n;
		K = k;
		spdlog::info("name:{} M:{}, N:{}, K:{}", name, M, N, K);
	}
	size_t mk() const {
		return M * K;
	}
	size_t kn() const {
		return K * N;
	}
	size_t mn() const {
		return M * N;
	}

	virtual void init(ElementAccumulator beta_ = ElementAccumulator(0)) = 0;
	virtual void profile() = 0;
	virtual double totalCompute() = 0;
	virtual float gflops(double runtime_ms) = 0;
	virtual bool checkResult() = 0;
	virtual ~BaseGEMMWrapper() = default;
	// I/O/Weight Setup
	virtual void set_weight(ElementInputB* data_b) = 0;
	virtual bool set_weight(vortexWeight& weight) = 0;
	virtual BaseGEMMWrapper& setOutput(pllmTensor<ElementOutput> data) = 0;
	virtual BaseGEMMWrapper& setA(pllmTensor<ElementInputA> data_a) = 0;
	virtual BaseGEMMWrapper& setB(pllmTensor<ElementInputB> data_b) = 0;
	virtual BaseGEMMWrapper& setC(pllmTensor<ElementOutput> data_c) = 0;
	virtual BaseGEMMWrapper& setD(pllmTensor<ElementOutput> data_d) = 0;
	virtual pllmTensor<ElementInputA> getA() = 0;
	virtual pllmTensor<ElementOutput> getD() = 0;
	virtual pllmTensor<ElementInputB> getB() = 0;
	virtual pllmTensor<ElementOutput> getC() = 0;
	virtual void set_alpha(float alpha_) = 0;
	virtual void set_beta(float beta_) = 0;
};

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
struct CutlassGEMMWrapper : public BaseGEMMWrapper {

	using ElementComputeEpilogue = ElementAccumulator;

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

	using EpilogueOp =
		cutlass::epilogue::thread::LinearCombination<ElementOutput,
													 128 /
														 cutlass::sizeof_bits<ElementOutput>::value,
													 ElementAccumulator,
													 ElementComputeEpilogue>;

	constexpr static bool isSplit = split_k > 1;
	constexpr static int stage_ = stages;
	using Gemm = typename cutlass::gemm::device::Gemm<ElementInputA,
													  LayoutInputA,
													  ElementInputB,
													  LayoutInputB,
													  ElementOutput,
													  LayoutOutput,
													  ElementAccumulator,
													  MMAOp,
													  SmArch,
													  ShapeMMAThreadBlock,
													  ShapeMMAWarp,
													  ShapeMMAOp,
													  EpilogueOp,
													  SwizzleThreadBlock,
													  stage_,
													  8,
													  8,
													  isSplit>;

	cutlass::TensorRef<ElementInputA, LayoutInputA> tensor_a_ref;
	cutlass::TensorRef<ElementInputB, LayoutInputB> tensor_b_ref;
	cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_c_ref;
	cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_d_ref;

	pllmTensor<ElementInputA> pllm_tensor_a;
	pllmTensor<ElementInputB> pllm_tensor_b;
	pllmTensor<ElementOutput> pllm_tensor_c;
	pllmTensor<ElementOutput> pllm_tensor_d;


	uint8_t* workspace;

	using TensorRefA = cutlass::TensorRef<ElementInputA, LayoutInputA>;
	using TensorRefB = cutlass::TensorRef<ElementInputB, LayoutInputB>;
	using TensorRefC = cutlass::TensorRef<ElementOutput, LayoutOutput>;
	using TensorRefD = cutlass::TensorRef<ElementOutput, LayoutOutput>;

	const size_t& kLda = std::is_same_v<LayoutInputA, cutlass::layout::RowMajor> ? K : M;
	const size_t& kLdb = std::is_same_v<LayoutInputB, cutlass::layout::RowMajor> ? N : K;
	const size_t& kLdc = std::is_same_v<LayoutOutput, cutlass::layout::RowMajor> ? N : M;
	const size_t& kLdd = kLdc;

	ElementComputeEpilogue alpha, beta;
	Gemm gemm_op;
	constexpr static size_t smem_size = sizeof(typename Gemm::GemmKernel::SharedStorage);
	bool inited = false;

	CutlassGEMMWrapper()
		: problem_size(M, N, K)
		, tensor_a_ref()
		, tensor_b_ref()
		, tensor_c_ref()
		, tensor_d_ref()
		, alpha(1)
		, beta(0) { }

	void work() {
		gemm_op(stream);
	}

	// Set both C and D to be the given buffer
	CutlassGEMMWrapper& setOutput(pllmTensor<ElementOutput> data) override {
		return setD(data);
	}

	// D = A * B + C
	CutlassGEMMWrapper& setA(pllmTensor<ElementInputA> data_a) override {
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
	CutlassGEMMWrapper& setB(pllmTensor<ElementInputB> data_b) override {
		if (std::is_same_v<LayoutInputB, cutlass::layout::RowMajor>) {
			assert(data_b.layout == PllmLayout::ROW_MAJOR);
			assert(data_b.dim1 == K);
			assert(data_b.dim2 == N);
		} else {
			assert(data_b.layout == PllmLayout::COL_MAJOR);
			assert(data_b.dim1 == N);
			assert(data_b.dim2 == K);
		}
		tensor_b_ref = TensorRefB(data_b.ptr, kLdb);
		pllm_tensor_b = data_b;
		return *this;
	}
	CutlassGEMMWrapper& setC(pllmTensor<ElementOutput> data_c) override {
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
	CutlassGEMMWrapper& setD(pllmTensor<ElementOutput> data_d) override {
		if (std::is_same_v<LayoutOutput, cutlass::layout::RowMajor>) {
			assert(data_d.layout == PllmLayout::ROW_MAJOR);
			assert(data_d.dim1 == M);
			assert(data_d.dim2 == N);
		} else {
			assert(data_d.layout == PllmLayout::COL_MAJOR);
			assert(data_d.dim1 == N);
			assert(data_d.dim2 == M);
		}
		tensor_d_ref = TensorRefD(data_d.ptr, kLdd);
		pllm_tensor_d = data_d;
		return *this;
	}

	pllmTensor<ElementInputA> getA() override {
		return pllm_tensor_a;
	}
	pllmTensor<ElementOutput> getD() override {
		return pllm_tensor_d;
	}

	pllmTensor<ElementInputB> getB() override {
		return pllm_tensor_b;
	}

	pllmTensor<ElementOutput> getC() override {
		return pllm_tensor_c;
	}

	inline bool isInitialized() const {
		return (problem_size.m() != 0);
	}

	// Expected usage:
	// Call set{A,B,C,D} to configure the input/output tensors before calling this init.
	// Assuming all tensor operands are setup.
	void init(ElementComputeEpilogue beta_) override {
		problem_size = cutlass::gemm::GemmCoord({int(M), int(N), int(K)});
		spdlog::info("name:{} M:{}, N:{}, K:{}, a, b, c, d: {}, {}, {}, {}", name, M, N, K, (size_t)tensor_a_ref.data(),  (size_t)tensor_b_ref.data(),  (size_t)tensor_c_ref.data(),  (size_t)tensor_d_ref.data());
		spdlog::info("lda, ldb, ldc, ldd: {}, {}, {}, {}", kLda, kLdb, kLdc, kLdd);
		beta = beta_;
		typename Gemm::Arguments arguments{problem_size,
										   tensor_a_ref,
										   tensor_b_ref,
										   tensor_c_ref,
										   tensor_d_ref,
										   {alpha, beta},
										   split_k};
		size_t workspace_size = Gemm::get_workspace_size(arguments);
		cudaMalloc(&workspace, workspace_size);
		cutlass::Status status = gemm_op.can_implement(arguments);
		CUTLASS_CHECK(status);
		status = gemm_op.initialize(arguments, workspace);
		CUTLASS_CHECK(status);
		inited = true;
	}

	void set_weight(ElementInputB* data_b) {

		tensor_b_ref = cutlass::TensorRef<ElementInputB, LayoutInputB>(data_b, kLdb);
		// Only update the gemm_op if we have already initialized this GEMM
		if(!inited) return; // kan: isInitilized not work if nrank != vnrank. use this inited can temporarily solve this problem.
		typename Gemm::Arguments arguments{problem_size,
										   tensor_a_ref,
										   tensor_b_ref,
										   tensor_c_ref,
										   tensor_d_ref,
										   {alpha, beta},
										   split_k};
		cutlass::Status status;
		// cutlass::Status status = gemm_op.can_implement(arguments);
		// CUTLASS_CHECK(status);
		status = gemm_op.update(arguments, workspace);
		CUTLASS_CHECK(status);
		pllm_tensor_b = pllmTensor<ElementInputB>(data_b, K, N, PllmLayout::ROW_MAJOR);
	}

	void set_alpha(float alpha_) override {
		alpha = alpha_;
		if(!inited) return; // kan: isInitilized not work if nrank != vnrank. use this inited can temporarily solve this problem.
		typename Gemm::Arguments arguments{problem_size,
										   tensor_a_ref,
										   tensor_b_ref,
										   tensor_c_ref,
										   tensor_d_ref,
										   {alpha, beta},
										   split_k};
		cutlass::Status status;
		status = gemm_op.update(arguments, workspace);
		CUTLASS_CHECK(status);
	}

	void set_beta(float beta_) {
		beta = beta_;
		if(!inited) return; // kan: isInitilized not work if nrank != vnrank. use this inited can temporarily solve this problem.
		typename Gemm::Arguments arguments{problem_size,
										   tensor_a_ref,
										   tensor_b_ref,
										   tensor_c_ref,
										   tensor_d_ref,
										   {alpha, beta},
										   split_k};
		cutlass::Status status;
		status = gemm_op.update(arguments, workspace);
		CUTLASS_CHECK(status);
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
		cutlass::reference::device::Gemm<ElementInputA,
										 LayoutInputA,
										 ElementInputB,
										 LayoutInputB,
										 ElementOutput,
										 LayoutOutput,
										 ElementComputeEpilogue,
										 ElementComputeEpilogue>
			gemm_device;
		cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d_standard(problem_size.mn());
		tensor_d_standard.sync_device();
		gemm_device(problem_size,
					alpha,
					tensor_a_ref,
					tensor_b_ref,
					beta,
					tensor_c_ref,
					tensor_d_standard.device_ref());
		cudaDeviceSynchronize();
		tensor_d_standard.sync_host();

		ElementOutput* data_d = new ElementOutput[M * N];
		cudaMemcpy(
			data_d, tensor_d_ref.data(), sizeof(ElementOutput) * M * N, cudaMemcpyDeviceToHost);

		bool passed = true;

		for(size_t i = 0; i < M * N; i++) {
			if((abs(data_d[i] - tensor_d_standard.host_data()[i]) -0.01) / abs(tensor_d_standard.host_data()[i]) > 1e-1) {
				passed = false;
				spdlog::error("i: {}, d: {}, standard: {}", i,
						static_cast<double>(data_d[i]),
						static_cast<double>(tensor_d_standard.host_data()[i]));
				break;
			}
		}

		return passed;
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
		std::cout << "warp: " << warp_m << " " << warp_n << " " << warp_k << std::endl;
		std::cout << "split: " << split_k << std::endl;
		std::cout << "stages: " << stage_ << std::endl;
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


	OperatorWrapper& logImpl(std::shared_ptr<spdlog::logger> logger = default_logger) override{
		log_tensor(logger, name+" input", pllm_tensor_a, 10, 20);
		log_tensor(logger, name+" weight", pllm_tensor_b, 10, 20);
		if (pllm_tensor_c.dim1 != 0 && pllm_tensor_c.dim2 != 0) {
			log_tensor(logger, name+" C", pllm_tensor_c, 10, 20);
		}
		else{
			logger->info("C is not set");
		}
		log_tensor(logger, name+" output", pllm_tensor_d, 10, 20);
		return *this;
	}

};
