// fast_uring.cpp  –  io_uring → Torch tensor using 2 MB huge‑page buffer

#include <fcntl.h>
#include <liburing.h>
#include <sys/mman.h>
#include <unistd.h>

#include <thread>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <chrono>

#include <torch/extension.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using  u64   = std::uint64_t;

constexpr std::size_t BLOCK      = 16 * 1024 * 1024;   // 16 MiB read chunk
constexpr unsigned     QD        = 64;                 // inflight SQEs
constexpr std::size_t  HUGEPAGE  = 2 * 1024 * 1024;    // 2 MB huge page

static inline void chk(int ret, const char* where)
{
    if (ret < 0) throw std::runtime_error(std::string(where) + ": " + strerror(-ret));
}

// ───────────────────────── per‑thread worker ────────────────────────────────
struct Worker {
    int   fd;
    char* base;
    u64   begin_blk;
    u64   nblocks;

    void operator()() {
        io_uring ring{};
        chk(io_uring_queue_init(QD, &ring, 0), "io_uring_queue_init");

        u64 blk = 0;
        while (blk < nblocks) {
            unsigned batch = std::min<u64>(QD, nblocks - blk);
            for (unsigned i = 0; i < batch; ++i) {
                auto* sqe = io_uring_get_sqe(&ring);
                void* dest = base + (blk + i) * BLOCK;
                off_t off  = (begin_blk + blk + i) * BLOCK;
                io_uring_prep_read(sqe, fd, dest, BLOCK, off);
            }
            io_uring_submit(&ring);

            unsigned done = 0;
            while (done < batch) {
                io_uring_cqe* cqe;
                chk(io_uring_wait_cqe(&ring, &cqe), "wait_cqe");
                if (cqe->res < 0)
                    throw std::runtime_error("read error: " + std::to_string(cqe->res));
                io_uring_cqe_seen(&ring, cqe);
                ++done;
            }
            blk += batch;
        }
        io_uring_queue_exit(&ring);
    }
};

// ───────────────────────── huge‑page allocator ──────────────────────────────
void* alloc_huge(u64 bytes)
{
#ifdef MAP_HUGETLB
    // round up to multiple of 2 MB
    u64 len = (bytes + HUGEPAGE - 1) & ~(HUGEPAGE - 1);

    // Some kernels require explicit huge‑page size flag (2 MB = log2(2 MB) ‑ MAP_HUGE_SHIFT)
#ifdef MAP_HUGE_2MB
    int flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB;
#else
    int flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB;
#endif
    void* p = mmap(nullptr, len, PROT_READ | PROT_WRITE, flags, -1, 0);
    if (p == MAP_FAILED) return nullptr;
    return p;
#else
    return nullptr;                       // platform lacks MAP_HUGETLB
#endif
}

// custom deleter for from_blob
void munmap_deleter(void* ptr, u64 bytes)
{
    if (ptr) munmap(ptr, bytes);
}

// ───────────────────────── top‑level loader ─────────────────────────────────
torch::Tensor load_fp16(const std::string& path,
                        unsigned threads    = 8,
                        u64      block_sz   = BLOCK,
                        bool     use_huge   = true)
{
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("open(): " + std::string(strerror(errno)));

    const u64 size    = ::lseek(fd, 0, SEEK_END);
    if (size % 2) throw std::runtime_error("file size not multiple of 2 bytes (fp16)");
    const u64 n_elem  = size / 2;
    const u64 nblocks = (size + block_sz - 1) / block_sz;

    // 1. allocate backing store ------------------------------------------------
    void* buf = nullptr;
    bool  huge_ok = false;

    if (use_huge) {
        buf = alloc_huge(size);
        huge_ok = (buf != nullptr);
        if (huge_ok)
            std::cerr << "[fast_uring] huge‑page mmap succeeded (" << (size >> 20)
                      << " MiB)\n";
        else
            std::cerr << "[fast_uring] huge‑page mmap failed – falling back to malloc\n";
    }
    if (!buf) {  // normal allocation fallback
        buf = std::aligned_alloc(64, size);   // 64‑byte aligned
        if (!buf) throw std::bad_alloc();
    }

    // 2. wrap in Torch tensor --------------------------------------------------
    auto deleter = [bytes=size, huge_ok](void* p) {
        if (huge_ok) munmap(p, (bytes + HUGEPAGE - 1) & ~(HUGEPAGE - 1));
        else         std::free(p);
    };

    auto options = torch::dtype(torch::kFloat16);
    torch::Tensor tensor = torch::from_blob(
                                buf,
                                {static_cast<long long>(n_elem)},
                                deleter,
                                options);

    char* base_ptr = static_cast<char*>(buf);

    // 3. launch worker threads -------------------------------------------------
    u64 per_thr = (nblocks + threads - 1) / threads;
    std::vector<std::thread> pool;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (unsigned t = 0; t < threads; ++t) {
        u64 sblk = t * per_thr;
        if (sblk >= nblocks) break;
        u64 blk_here = std::min<u64>(per_thr, nblocks - sblk);
        char* slice  = base_ptr + sblk * block_sz;
        pool.emplace_back(Worker{fd, slice, sblk, blk_here});
    }
    for (auto& th : pool) th.join();
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cerr << "[fast_uring] Loaded " << (size >> 20) << " MiB in "
              << ms << " ms → " << (size / 1e6 / (ms/1000.0)) << " MB/s\n";

    ::close(fd);
    return tensor;
}

// ───────────────────────── pybind module glue ───────────────────────────────
PYBIND11_MODULE(fast_uring, m) {
    m.doc() = "io_uring reader into a huge‑page Torch tensor";
    m.def("load_fp16", &load_fp16,
          py::arg("path"),
          py::arg("threads")  = 32,
          py::arg("block")    = BLOCK,
          py::arg("use_huge") = true,
          R"pbdoc(
Read a raw little‑endian fp16 file into a 1‑D torch tensor.

If `use_huge` is true the code first tries to mmap the destination buffer
with `MAP_HUGETLB` (2 MB pages).  You must pre‑reserve huge pages with
`vm.nr_hugepages` (size / 2 MB) for this to succeed; otherwise it falls
back to standard heap allocation without aborting.

Parameters
----------
path : str
threads : int, default 8
block : int, default 16 MiB
use_huge : bool, default True

Returns
-------
torch.Tensor(fp16)
)pbdoc");
}
