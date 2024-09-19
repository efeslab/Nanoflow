#pragma once
#include <span>
#include <concepts>
#include <cassert>
#include "tensor.cuh"
#include "spdlog/spdlog.h"

template <typename Tdst, typename Tsrc>
inline Tdst type_cast(Tsrc src) {
    static_assert(sizeof(Tdst) == sizeof(Tsrc));
    return reinterpret_cast<Tdst>(src);
}

template <typename Tdst, typename Tsrc>
inline Tdst* ptr_cast(Tsrc* src) {
    static_assert(sizeof(Tdst) == sizeof(Tsrc));
    return reinterpret_cast<Tdst*>(src);
}

template <typename Tsrc, typename Tdst, std::size_t Extent = std::dynamic_extent>
inline std::span<Tdst, Extent> span_cast(std::span<Tsrc, Extent> src) {
    static_assert(sizeof(Tdst) == sizeof(Tsrc));
    return std::span<Tdst, Extent>{reinterpret_cast<Tdst*>(src.data()), src.size()};
}

template <typename T>
class AllocationManager {
public:
	// size is the number of element of type T (not in bytes)
	AllocationManager(T* base, size_t size)
		: base(base)
		, allocated(0)
		, size(size) { }

	T* alloc(size_t n) {
		T* ret = base + allocated;
		assert(allocated + n <= size);
		allocated += n;
		return ret;
	}

	std::span<T> allocSpan(size_t size) {
		return std::span{alloc(size), size};
	}

	pllmTensor<T> allocTensor(int M, int N, PllmLayout layout) {
		if (layout == PllmLayout::ROW_MAJOR)
			return pllmTensor<T>{alloc(M * N), M, N, layout};
		else
			return pllmTensor<T>{alloc(M * N), N, M, layout};
	}

	size_t getAllocation(){
		return allocated;
	}

private:
	T* base;
	size_t allocated;
	size_t size;
};

template<typename T>
inline std::span<T> getSubSpan(const std::span<T>& span, int rank, int nranks) {
    assert(span.size() % nranks == 0);
    int chunk_size = span.size() / nranks;
    return std::span{span.data() + rank * chunk_size, chunk_size};
}

template <typename T, std::integral... Lens>
inline std::array<std::span<T>, sizeof...(Lens)> splitSpan(const std::span<T>& span_in, Lens... lens) {
	assert((span_in.size() == (0 + ... + lens)) &&
		   "Subspan length sum mismatches with the input span");
	size_t i = 0;
	size_t offset = 0;
	std::array<std::span<T>, sizeof...(Lens)> ret;
	((ret[i++] = std::span{span_in.data() + offset, static_cast<size_t>(lens)}, offset += lens),
	 ...);
	return ret;
}

// Split a single span to subspans according to given list of lengths.
// @param suffix means the suffix appened to each subspan that overlaps with the next subspan. 0 means non-overlaping
// Example, splitSpan({1,2,3,4}, {1,2}, 1) => {1,2} (2 is the overlapping suffix), {2,3,4} (4 is the suffix);
template <typename T,
		  size_t N,
		  std::unsigned_integral TLen,
		  template <class, size_t>
		  class LensContainer>
inline std::array<std::span<T>, N>
splitSpan(const std::span<T>& span_in, const LensContainer<TLen, N>& lens, TLen suffix = 0) {
	assert((span_in.size() == std::accumulate(lens.begin(), lens.end(), 0) + suffix) &&
		   "Subspan length sum mismatches with the input span");
	size_t i = 0;
	size_t offset = 0;
	std::array<std::span<T>, N> ret;
	for(size_t i = 0; i < N; ++i) {
		size_t l = lens[i];
		ret[i] = std::span{span_in.data() + offset, l + suffix};
		offset += l;
	}
	return ret;
}