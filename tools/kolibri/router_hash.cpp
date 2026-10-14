// Kolibri 1's router over a fixed synthetic input: every logit hashed and the
// launch timed, per width.
//
// Two questions about the two router shapes in
// `src/models/kolibri1/kernels/rocm/route.hip.cpp`: does the tiled one publish
// the logits the narrow one does — it has to, and the hash of all 384 of them
// at every width is the proof — and where the two cross over in cost. The
// launch picks its shape by `kRouterTiledMinRows`, so a crossover measurement
// builds this twice, once with that constant at one and once past the widest
// width swept here. The hashes of the two builds agree, which is what makes
// that a timing experiment rather than a numerics one.
//
// Not part of the build and no target. Inside the build container, against a
// configured tree:
//   c++ -std=c++20 -O2 -D__HIP_PLATFORM_AMD__ -I/work -I/opt/rocm/include \
//       tools/kolibri/router_hash.cpp \
//       build/gpu-test/src/models/kolibri1/libgufo_kolibri1.a \
//       build/gpu-test/libgufo_core.a \
//       -L/opt/rocm/lib -lamdhip64 -lcrypto -licui18n -licuuc -pthread \
//       -o artifacts/kolibri/router_hash
// and then on the host that has the render node:
//   LD_LIBRARY_PATH=/opt/rocm/lib artifacts/kolibri/router_hash
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdint>
#include <vector>

#include "src/models/kolibri1/kernels/rocm/moe.hpp"

// `core/hip/hip_utils.hpp` is where this macro lives, but it wants
// -DENGINE_ENABLE_HIP and hipblas; a measurement tool can carry its own.
#define HIP_CHECK(call)                                                   \
  do {                                                                    \
    const hipError_t err_ = (call);                                       \
    if (err_ != hipSuccess) {                                             \
      std::fprintf(stderr, "HIP error %s in %s:%d (%s)\n",                \
                   hipGetErrorString(err_), __FILE__, __LINE__, #call);    \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

namespace k1 = gufo::models::kolibri1::rocm;

namespace {

struct Lcg {
  std::uint64_t s{0x2545F4914F6CDD1DULL};
  float next() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    const auto u = static_cast<std::uint32_t>(s >> 32);
    return (static_cast<float>(u) / 4294967295.0F) * 2.0F - 1.0F;
  }
};

template <typename T>
T* Alloc(std::size_t bytes) {
  void* p = nullptr;
  HIP_CHECK(hipMalloc(&p, bytes));
  return static_cast<T*>(p);
}

std::uint64_t Fnv(const unsigned char* data, std::size_t n) {
  std::uint64_t h = 1469598103934665603ULL;
  for (std::size_t i = 0; i < n; ++i) {
    h ^= data[i];
    h *= 1099511628211ULL;
  }
  return h;
}

}  // namespace

int main() {
  constexpr std::uint32_t kHidden = 2560;
  constexpr std::uint32_t kExperts = 384;
  constexpr std::uint32_t kUsed = 6;
  constexpr std::uint32_t kMaxRows = 256;

  std::vector<float> host_x(static_cast<std::size_t>(kMaxRows) * kHidden);
  std::vector<float> host_router(static_cast<std::size_t>(kExperts) * kHidden);
  std::vector<float> host_bias(kExperts);
  Lcg lcg;
  for (auto& v : host_x) v = lcg.next() * 0.02F;
  for (auto& v : host_router) v = lcg.next() * 0.02F;
  for (auto& v : host_bias) v = lcg.next() * 0.5F;

  float* x = Alloc<float>(host_x.size() * sizeof(float));
  float* router = Alloc<float>(host_router.size() * sizeof(float));
  float* bias = Alloc<float>(host_bias.size() * sizeof(float));
  // `Alloc` takes bytes. Sized in elements these were too small from 128 rows
  // up, which is what used to smash the allocator.
  float* logits = Alloc<float>(static_cast<std::size_t>(kMaxRows) * kExperts *
                               sizeof(float));
  std::int32_t* ids = Alloc<std::int32_t>(static_cast<std::size_t>(kMaxRows) *
                                          kUsed * sizeof(std::int32_t));
  float* weights = Alloc<float>(static_cast<std::size_t>(kMaxRows) * kUsed *
                                sizeof(float));
  std::uint32_t* counts =
      Alloc<std::uint32_t>(kExperts * sizeof(std::uint32_t));
  std::int32_t* groups =
      Alloc<std::int32_t>(k1::GroupTableInts(kExperts) * sizeof(std::int32_t));
  HIP_CHECK(hipMemcpy(x, host_x.data(), host_x.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(router, host_router.data(),
                      host_router.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(bias, host_bias.data(), host_bias.size() * sizeof(float),
                      hipMemcpyHostToDevice));

  hipEvent_t begin;
  hipEvent_t end;
  HIP_CHECK(hipEventCreate(&begin));
  HIP_CHECK(hipEventCreate(&end));

  for (const std::uint32_t rows :
       {1U, 2U, 4U, 8U, 16U, 32U, 64U, 128U, 256U}) {
    const std::size_t bytes =
        static_cast<std::size_t>(rows) * kExperts * sizeof(float);
    for (int rep = 0; rep < 5; ++rep) {
      HIP_CHECK(hipMemsetAsync(counts, 0, kExperts * sizeof(std::uint32_t)));
      k1::MoeRouteArgs a;
      a.x = x;
      a.router = router;
      a.bias = bias;
      a.logits = logits;
      a.ids = ids;
      a.weights = weights;
      a.counts = counts;
      a.groups = groups;
      a.rows = rows;
      a.hidden = kHidden;
      a.experts = kExperts;
      a.used = kUsed;
      HIP_CHECK(hipEventRecord(begin));
      k1::MoeRoute(a, 0);
      HIP_CHECK(hipEventRecord(end));
      HIP_CHECK(hipEventSynchronize(end));
      const hipError_t le = hipGetLastError();
      if (le != hipSuccess) {
        std::fprintf(stderr, "  rows=%u rep=%d fehler: %s\n", rows, rep,
                     hipGetErrorString(le));
      }
      float ms = 0.0F;
      HIP_CHECK(hipEventElapsedTime(&ms, begin, end));
      if (rep == 4) {
        std::vector<float> back(rows * kExperts);
        HIP_CHECK(hipMemcpy(back.data(), logits, bytes, hipMemcpyDeviceToHost));
        const auto h = Fnv(reinterpret_cast<const unsigned char*>(back.data()),
                           bytes);
        std::printf("rows=%3u  %8.1f us/launch  logits_hash=%016llx  "
                    "l[0]=%.9g l[1]=%.9g l[last]=%.9g\n",
                    rows, ms * 1000.0F / 1, static_cast<unsigned long long>(h),
                    back[0], back[1], back[back.size() - 1]);
      }
    }
  }
  return 0;
}
