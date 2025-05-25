#pragma once

#include "cutlassGemmBase.cuh"
using ColumnMajor = cutlass::layout::ColumnMajor;
using RowMajor = cutlass::layout::RowMajor;
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
struct CutlassGEMMWrapper : public BaseGEMMWrapperTemplate<LayoutInputA_, LayoutInputB_, LayoutOutput_> {
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






	uint8_t* workspace;


	ElementComputeEpilogue alpha, beta;
	Gemm gemm_op;
	constexpr static size_t smem_size = sizeof(typename Gemm::GemmKernel::SharedStorage);
	bool inited = false;

	CutlassGEMMWrapper()
		: problem_size(int(M), int(N), int(K))
		, alpha(1)
		, beta(0) { }

	void work() {
		if (M !=0) {
			gemm_op(stream);
		}
		else {
			// spdlog::error("GEMM {} batch size is 0", name);
		}
	}

	// Set both C and D to be the given buffer


	// D = A * B + C

	// CutlassGEMMWrapper& setB(pllmTensor<ElementInputB> data_b) override {
	// 	if (std::is_same_v<LayoutInputB, cutlass::layout::RowMajor>) {
	// 		assert(data_b.layout == PllmLayout::ROW_MAJOR);
	// 		assert(data_b.dim1 == K);
	// 		assert(data_b.dim2 == N);
	// 	} else {
	// 		assert(data_b.layout == PllmLayout::COL_MAJOR);
	// 		assert(data_b.dim1 == N);
	// 		assert(data_b.dim2 == K);
	// 	}
	// 	tensor_b_ref = TensorRefB(data_b.ptr, kLdb);
	// 	pllm_tensor_b = data_b;
	// 	return *this;
	// }



	inline bool isInitialized() const {
		return (problem_size.m() != 0);
	}
	void updateArgument(){
		problem_size = cutlass::gemm::GemmCoord({M, N, K});
		typename Gemm::Arguments arguments{problem_size,
										   tensor_a_ref,
										   tensor_b_ref,
										   tensor_c_ref,
										   tensor_d_ref,
										   {alpha, beta},
										   split_k};
		cutlass::Status status = gemm_op.initialize(arguments, workspace);
		CUTLASS_CHECK(status);
	}
	// Expected usage:
	// Call set{A,B,C,D} to configure the input/output tensors before calling this init.
	// Assuming all tensor operands are setup.
	void init(ElementComputeEpilogue beta_) override {
		problem_size = cutlass::gemm::GemmCoord({int(M), int(N), int(K)});
		// spdlog::info("name:{} M:{}, N:{}, K:{}, a, b, c, d: {}, {}, {}, {}", name, M, N, K, (size_t)tensor_a_ref.data(),  (size_t)tensor_b_ref.data(),  (size_t)tensor_c_ref.data(),  (size_t)tensor_d_ref.data());
		// spdlog::info("lda, ldb, ldc, ldd: {}, {}, {}, {}", kLda, kLdb, kLdc, kLdd);
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
				// spdlog::error("i: {}, d: {}, standard: {}", i,
				// 		static_cast<double>(data_d[i]),
				// 		static_cast<double>(tensor_d_standard.host_data()[i]));
				break;
			}
		}

		return passed;
	}

	float gflops(double runtime_ms) {
		return float(problem_size.product()) * 2.0f / 1e6f / float(runtime_ms);
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

};
