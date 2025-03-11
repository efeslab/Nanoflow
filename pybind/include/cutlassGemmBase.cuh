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
		// spdlog::info("name:{} M:{}, N:{}, K:{}", name, M, N, K);
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
	virtual BaseGEMMWrapper& setA(ElementInputA* data_a) = 0;
	// virtual BaseGEMMWrapper& setB(ElementInputB* data_b) = 0;
	virtual BaseGEMMWrapper& setC(ElementOutput* data_c) = 0;
	virtual BaseGEMMWrapper& setD(ElementOutput* data_d) = 0;
	virtual ElementInputA* getA() = 0;
	virtual ElementInputB* getB() = 0;
	virtual ElementOutput* getC() = 0;
	virtual ElementOutput* getD() = 0;
	virtual void set_alpha(float alpha_) = 0;
	virtual void set_beta(float beta_) = 0;
	virtual void updateArgument() = 0;

};
template <typename LayoutInputA_ = cutlass::layout::ColumnMajor,
		  typename LayoutInputB_ = cutlass::layout::ColumnMajor,
		  typename LayoutOutput_ = cutlass::layout::ColumnMajor>
struct BaseGEMMWrapperTemplate : public BaseGEMMWrapper {

	ElementInputA* input_a;
	ElementInputB* input_b;
	ElementOutput* input_c;
	ElementOutput* output_d;

	using LayoutInputA = LayoutInputA_;
	using LayoutInputB = LayoutInputB_;
	using LayoutOutput = LayoutOutput_;

	using TensorRefA = cutlass::TensorRef<ElementInputA, LayoutInputA>;
	using TensorRefB = cutlass::TensorRef<ElementInputB, LayoutInputB>;
	using TensorRefC = cutlass::TensorRef<ElementOutput, LayoutOutput>;
	using TensorRefD = cutlass::TensorRef<ElementOutput, LayoutOutput>;

	cutlass::TensorRef<ElementInputA, LayoutInputA> tensor_a_ref;
	cutlass::TensorRef<ElementInputB, LayoutInputB> tensor_b_ref;
	cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_c_ref;
	cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_d_ref;

	const size_t& kLda = std::is_same_v<LayoutInputA, cutlass::layout::RowMajor> ? K : M;
	const size_t& kLdb = std::is_same_v<LayoutInputB, cutlass::layout::RowMajor> ? N : K;
	const size_t& kLdc = std::is_same_v<LayoutOutput, cutlass::layout::RowMajor> ? N : M;
	const size_t& kLdd = kLdc;
	
	// OperatorWrapper& logImpl(std::shared_ptr<spdlog::logger> logger = default_logger) override{
	// 	log_tensor(logger, name+" input", input_a, 10, 20);
	// 	log_tensor(logger, name+" weight", input_b, 10, 20);
	// 	if (input_c.dim1 != 0 && input_c.dim2 != 0) {
	// 		log_tensor(logger, name+" C", input_c, 10, 20);
	// 	}
	// 	else{
	// 		logger->info("C is not set");
	// 	}
	// 	log_tensor(logger, name+" output", output_d, 10, 20);
	// 	return *this;
	// }
	ElementInputA* getA() {
		return input_a;
	}
	ElementOutput* getD() {
		return output_d;
	}

	ElementInputB* getB() {
		return input_b;
	}

	ElementOutput* getC() {
		return input_c;
	}

	BaseGEMMWrapper& setA(ElementInputA* data_a) {
		tensor_a_ref = TensorRefA(data_a, kLda);
		input_a = data_a;
		return *this;
	}
	BaseGEMMWrapper& setC(ElementOutput* data_c) override {
		tensor_c_ref = TensorRefC(data_c, kLdc);
		input_c = data_c;
		return *this;
	}
	BaseGEMMWrapper& setD(ElementOutput* data_d) override {
		tensor_d_ref = TensorRefD(data_d, kLdd);
		output_d = data_d;
		return *this;
	}
	
	virtual void updateArgument() = 0;
};