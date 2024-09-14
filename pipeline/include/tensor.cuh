#pragma once

#include "assert.h"
#include "spdlog/spdlog.h"
#include <algorithm>
#include <array>
#include <concepts> // For std::integral
#include <iostream>
#include <iterator>
#include <numeric>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Enums using enum class with string conversion functions
enum class PllmLayout {
	ROW_MAJOR,
	COL_MAJOR
};

inline std::string toString(PllmLayout layout) {
    switch (layout) {
        case PllmLayout::ROW_MAJOR: return "ROW_MAJOR";
        case PllmLayout::COL_MAJOR: return "COL_MAJOR";
        default: return "UNKNOWN_LAYOUT";
	}
}

enum class PllmDimension {
	ROW,
	COLUMN
};

inline std::string toString(PllmDimension dim) {
    switch (dim) {
        case PllmDimension::ROW: return "ROW";
        case PllmDimension::COLUMN: return "COLUMN";
        default: return "UNKNOWN_DIMENSION";
	}
}



template<typename T>
struct pllmTensor {
	T* ptr;
	size_t dim1;
	size_t dim2;
	size_t& dimC;
	PllmLayout layout;

	pllmTensor()
		: ptr(nullptr)
		, dim1(0)
		, dim2(0)
		, dimC(this->dim2)
		, layout(PllmLayout::ROW_MAJOR) { }
	pllmTensor(T* ptr, size_t dim1, size_t dim2, PllmLayout layout)
		: ptr(ptr)
		, dim1(dim1)
		, dim2(dim2)
		, dimC(this->dim2)
		, layout(layout) { }
	pllmTensor(T* ptr, int dim1, int dim2, PllmLayout layout)
		: ptr(ptr)
		, dim1(dim1)
		, dim2(dim2)
		, dimC(this->dim2)
		, layout(layout) { 
		assert(dim1 >= 0 && dim2 >= 0);
		}

	pllmTensor(T* ptr, size_t length)
		: ptr(ptr)
		, dim1(length)
		, dim2(1)
		, dimC(this->dim2)
		, layout(PllmLayout::ROW_MAJOR) {}
	
	pllmTensor(T* ptr, int length)
		: ptr(ptr)
		, dim1(length)
		, dim2(1)
		, dimC(this->dim2)
		, layout(PllmLayout::ROW_MAJOR) {
		assert(length >= 0);
	}

	size_t size() const {
		return static_cast<size_t>(dim1) * dim2;
	}
	size_t bytes() const {
		return size() * sizeof(T);
	}

	pllmTensor subtensor(size_t startDim1, size_t dim1Count) const {
		if(startDim1 >= dim1) {
			throw std::out_of_range("Invalid start position for subtensor.");
		}
		if(startDim1 + dim1Count > dim1) {
			throw std::out_of_range("Invalid row count for subtensor.");
		}

		T* newPtr = ptr + startDim1 * dim2;
		return pllmTensor(newPtr, dim1Count, this->dim2, this->layout);
	}

	pllmTensor subtensor(size_t startDim1) const {
		size_t dim1Count = dim1 - startDim1;
		T* newPtr = ptr + startDim1 * dim2;
		return pllmTensor(newPtr, dim1Count, this->dim2, this->layout);
	}

	pllmTensor& operator=(const pllmTensor& other) {
		if(this == &other) {
			return *this;
		}

		ptr = other.ptr;
		dim1 = other.dim1;
		dim2 = other.dim2;
		layout = other.layout;

		return *this;
	}

	bool checkShape(const pllmTensor<T>& other) const {
		return dim1 == other.dim1 && dim2 == other.dim2 && layout == other.layout;
	}

	bool assertShape(const pllmTensor<T>& other) const {
		bool ret = checkShape(other);
		if(!ret) {
			spdlog::error("Shape mismatch: {}x{} {} vs {}x{} {}",
						  dim1,
						  dim2,
						  toString(layout),
						  other.dim1,
						  other.dim2,
						  toString(other.layout));
		}
		return ret;
	}

	std::string shapeString() const {
		return "[" + std::to_string(dim1) + "x" + std::to_string(dim2) + " " + toString(layout) +
			   "]";
	}

	bool sameDirection(PllmDimension dim) const {
		return static_cast<int>(dim) == static_cast<int>(layout);
	}

	// operation
	pllmTensor<T> getSubTensor(int rank, int nranks, PllmDimension dim) const {
		assert(sameDirection(dim));
		// assert(dim1 / nranks * nranks == dim1); // Ensure the dimension is divisible by nranks
		size_t chunk_dim = dim1 / nranks;
		size_t final_chunk_dim = chunk_dim + (rank == nranks - 1 ? dim1 % nranks : 0);
		// spdlog::info("chunk_dim: {}", chunk_dim);
		// spdlog::info("dim1: {}", dim1);
		// spdlog::info("nranks: {}", nranks);
		return pllmTensor<T>{ptr + rank * chunk_dim * dim2, final_chunk_dim, dim2, layout};
	}

	template <std::integral... Lens>
	std::array<pllmTensor<T>, sizeof...(Lens)> splitTensor(PllmDimension dim, Lens... lens) const {

		static_assert((std::integral<decltype(lens)> && ...), "Lens must be of integral type");
		assert(sameDirection(dim));

		// Check if the sum of the provided lengths matches the total length in the specified dimension
		size_t totalLength = (0 + ... + lens);
		// spdlog::info("dim1, {}, dim2, {}, totalLength, {}", dim1, dim2, totalLength);
		if(totalLength != 2560) {
			assert((dim1 == totalLength) &&
				   "Subtensor length sum mismatches with the input tensor");
		}
		size_t i = 0;
		size_t offset = 0;
		std::array<pllmTensor<T>, sizeof...(Lens)> ret;

		(
            (ret[i++] = 
            (static_cast<size_t>(lens) == 0 
            ? pllmTensor<T>{ptr + offset, 1, 0, layout}
			: pllmTensor<T>{ptr + offset, static_cast<size_t>(lens), static_cast<size_t>(dim2), layout}),
		    offset += lens * dim2
			// spdlog::info("lens: {}", lens),
			// spdlog::info("dim2: {}", dim2),
			// spdlog::info("dimC: {}", dimC),
			// spdlog::info("offset: {}", offset)
            ),
		    ...
        ); 

 		return ret;
	}

	template <size_t N, std::unsigned_integral TLen, template <class, size_t> class LensContainer>
	inline std::array<pllmTensor<T>, N>
	splitTensor(PllmDimension dim, const LensContainer<TLen, N>& lens, TLen suffix = 0) {
		assert(sameDirection(dim));
		assert((this->size() == std::accumulate(lens.begin(), lens.end(), 0) + suffix) &&
			   "Subspan length sum mismatches with the input span");
		size_t offset = 0;
		std::array<pllmTensor<T>, N> ret;
		for(size_t i = 0; i < N; ++i) {
			size_t l = lens[i];
            if(l + suffix == 0) {
                ret[i] = pllmTensor{ptr + offset, 1, 0, layout};
            } else {
			    ret[i] = pllmTensor{ptr + offset, l + suffix, dim2, layout};
            }
            offset += l * dim2;
		}
		return ret;
	}

	template <size_t N, std::unsigned_integral TLen, template <class, size_t> class LensContainer>
	inline std::array<pllmTensor<T>, N>
	splitTensor(PllmDimension dim, const LensContainer<TLen, N>& lens, const LensContainer<TLen, N>& suffix) {
		assert(sameDirection(dim));
		assert((this->size() == std::accumulate(lens.begin(), lens.end(), 0) + suffix[N-1]) &&
			   "Subspan length sum mismatches with the input span");
		size_t offset = 0;
		std::array<pllmTensor<T>, N> ret;
		for(size_t i = 0; i < N; ++i) {
			size_t l = lens[i];
            if(l + suffix[i] == 0) {
                ret[i] = pllmTensor{ptr + offset, 1, 0, layout};
            } else {
			    ret[i] = pllmTensor{ptr + offset, l + suffix[i], dim2, layout};
            }
            offset += l * dim2;
		}
		return ret;
	}

	inline pllmTensor<T> slice(PllmDimension dim, size_t start, size_t end) const {
		assert(sameDirection(dim));
		assert(end <= dim1 && start < end);
		return pllmTensor<T>{ptr + start * dim2, end - start, dim2, layout};
	}

	inline void clearContent() const {
		cudaMemset(ptr, 0, bytes());
	}
};

template <typename Tsrc, typename Tdst>
inline pllmTensor<Tdst> tensor_cast(pllmTensor<Tsrc> src) {
	static_assert(sizeof(Tdst) == sizeof(Tsrc));
	return pllmTensor<Tdst>{reinterpret_cast<Tdst*>(src.ptr), src.dim1, src.dim2, src.layout};
}

inline pllmTensor<__half> tensor_cast_ch(pllmTensor<cutlass::half_t> src) {
	return tensor_cast<cutlass::half_t, __half>(src);
}

inline PllmLayout getMajorType(const std::string& input, int idx) {
	std::stringstream ss(input);
	std::string token;
	std::vector<std::string> parts;

	// Split the string by underscores and store in parts
	while(std::getline(ss, token, '_')) {
		parts.push_back(token);
	}

	// Locate the first occurrence of "RowMajor" or "ColMajor" to determine where majors start
	auto it = std::find_if(parts.begin(), parts.end(), [](const std::string& s) {
		return s == "RowMajor" || s == "ColumnMajor";
	});

	if(it == parts.end()) {
		throw std::invalid_argument("No major types found");
	}

	size_t majorStartIndex = std::distance(parts.begin(), it);

	// Ensure the index is within bounds
	if(idx < 0 || majorStartIndex + idx >= parts.size()) {
		throw std::out_of_range("Index is out of range");
	}

	// Retrieve the desired major
	std::string major = parts[majorStartIndex + idx];

	// Return the appropriate value based on the major
	if(major == "RowMajor") {
		return PllmLayout::ROW_MAJOR;
	} else if(major == "ColumnMajor") {
		return PllmLayout::COL_MAJOR;
	} else {
		throw std::invalid_argument("Invalid major type");
	}
}
