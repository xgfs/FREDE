#include <math.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace std;

#ifdef _WIN32
#define DLLEXPORT extern "C" __declspec(dllexport)
#else
#define DLLEXPORT extern "C"
#endif

#ifndef UINT64_C
#define UINT64_C(c) (c##ULL)
#endif

#define DEFAULT_ALIGN 128

typedef unsigned long long ull;

uint64_t rng_seed[2];

static inline uint64_t rotl(const uint64_t x, int k) {
  return (x << k) | (x >> (64 - k));
}

// http://xoroshiro.di.unimi.it/#shootout
uint64_t lrand() {
  const uint64_t s0 = rng_seed[0];
  uint64_t s1 = rng_seed[1];
  const uint64_t result = s0 + s1;
  s1 ^= s0;
  rng_seed[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14);  // a, b
  rng_seed[1] = rotl(s1, 36);                    // c
  return result;
}

static inline double drand() {
  const union un {
    uint64_t i;
    double d;
  } a = {UINT64_C(0x3FF) << 52 | lrand() >> 12};
  return a.d - 1.0;
}

inline void *aligned_malloc(size_t size, size_t align) {
#ifndef _MSC_VER
  void *result;
  if (posix_memalign(&result, align, size)) result = 0;
#else
  void *result = _aligned_malloc(size, align);
#endif
  return result;
}

inline void aligned_free(void *ptr) {
#ifdef _MSC_VER
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

inline int irand(int min, int max) { return lrand() % (max - min) + min; }

inline int irand(int max) { return lrand() % max; }

DLLEXPORT int ppr_row_matmul(float *x, const int *offsets, const int *edges,
                             const int *degrees, const int nv, const int source,
                             const float alpha = 0.85, const int n_threads = 1,
                             const double tol = 1e-6, const int max_iter = 100,
                             int schedule_size = 2048) {
  float *x_last =
      static_cast<float *>(aligned_malloc(nv * sizeof(float), DEFAULT_ALIGN));

  for (int i = 0; i < nv; i++) x[i] = 1.0 / nv;

  for (int iter = 0; iter < max_iter; iter++) {
#pragma omp parallel for schedule(static) num_threads(n_threads)
    for (int i = 0; i < nv; i++) x_last[i] = x[i];
#pragma omp parallel for schedule(static) num_threads(n_threads)
    for (int i = 0; i < nv; i++) x[i] = 0;
    x[source] += 1 - alpha;
#pragma omp parallel for schedule(nonmonotonic              \
                                  : dynamic, schedule_size) \
    num_threads(n_threads)
    for (int i = 0; i < nv; i++)
      for (int j = offsets[i]; j < offsets[i + 1]; j++)
#pragma omp atomic
        x[edges[j]] += alpha * x_last[i] / degrees[i];
    double error = 0;
#pragma omp simd reduction(+ : error)
    for (int i = 0; i < nv; i++) error += fabs(x[i] - x_last[i]);
    if (error < tol) break;
  }
  aligned_free(x_last);
  return x[0];
}

DLLEXPORT int ppr_mat_matmul(float *xmat, const int *offsets, const int *edges,
                             const int *degrees, const int nv,
                             const float alpha = 0.85, const int n_threads = 1,
                             const double tol = 1e-6, const int max_iter = 100,
                             int schedule_size = 2048) {
  float *x_last =
      static_cast<float *>(aligned_malloc(nv * sizeof(float), DEFAULT_ALIGN));
  for (int k = 0; k < nv; k++) {
    float *x = &xmat[k * nv];
    for (int i = 0; i < nv; i++) x[i] = 1.0 / nv;
    for (int iter = 0; iter < max_iter; iter++) {
#pragma omp parallel for schedule(static) num_threads(n_threads)
      for (int i = 0; i < nv; i++) x_last[i] = x[i];
#pragma omp parallel for schedule(static) num_threads(n_threads)
      for (int i = 0; i < nv; i++) x[i] = 0;
      x[k] += 1 - alpha;
#pragma omp parallel for schedule(nonmonotonic              \
                                  : dynamic, schedule_size) \
    num_threads(n_threads)
      for (int i = 0; i < nv; i++)
        for (int j = offsets[i]; j < offsets[i + 1]; j++)
#pragma omp atomic
          x[edges[j]] += alpha * x_last[i] / degrees[i];
      double error = 0;
#pragma omp simd reduction(+ : error)
      for (int i = 0; i < nv; i++) error += fabs(x[i] - x_last[i]);
      if (error < tol) break;
    }
  }
  aligned_free(x_last);
  return 0;
}

DLLEXPORT int ppr_row_rw(float *x, const int *offsets, const int *edges,
                         const int *degrees, const int nv, const int source,
                         const float alpha = 0.85, const int n_threads = 1,
                         const int nsamples = 1000000) {
  int *counts =
      static_cast<int *>(aligned_malloc(nv * sizeof(int), DEFAULT_ALIGN));
  memset(counts, 0, nv * sizeof(int));
#pragma omp parallel for
  for (int i = 0; i < nsamples; i++) {
    int current = source;
    while (drand() < alpha) {
      if (offsets[current] == offsets[current + 1]) break;
      current = edges[irand(offsets[current], offsets[current + 1])];
    }
#pragma omp atomic
    counts[current]++;
  }
#pragma omp simd
  for (int i = 0; i < nv; i++) x[i] = counts[i] / double(nsamples);
  aligned_free(counts);
  return 0;
}

DLLEXPORT int ppr_mat_rw(float *x, const int *offsets, const int *edges,
                         const int *degrees, const int nv,
                         const float alpha = 0.85, const int n_threads = 1,
                         const int nsamples = 100000) {
  int *counts =
      static_cast<int *>(aligned_malloc(nv * sizeof(int), DEFAULT_ALIGN));
  for (int j = 0; j < nv; j++) {
    memset(counts, 0, nv * sizeof(int));
#pragma omp parallel for reduction(+ : counts[:nv]) num_threads(n_threads)
    for (int i = 0; i < nsamples; i++) {
      int current = j;
      while (drand() < alpha) {
        if (offsets[current] == offsets[current + 1]) break;
        current = edges[irand(offsets[current], offsets[current + 1])];
      }
      counts[current]++;
    }
#pragma omp simd
    for (int i = 0; i < nv; i++) x[i + j * nv] = counts[i] / double(nsamples);
    cout << j << endl;
  }
  aligned_free(counts);
  return 0;
}

DLLEXPORT int seed_rand(int t) {
  for (int i = 0; i < 2; i++) {
    ull z = t += UINT64_C(0x9E3779B97F4A7C15);
    z = (z ^ z >> 30) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ z >> 27) * UINT64_C(0x94D049BB133111EB);
    rng_seed[i] = z ^ z >> 31;
  }
  return 0;
}

DLLEXPORT int init_rand() {
  ull t = time(nullptr);
  for (int i = 0; i < 2; i++) {
    ull z = t += UINT64_C(0x9E3779B97F4A7C15);
    z = (z ^ z >> 30) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ z >> 27) * UINT64_C(0x94D049BB133111EB);
    rng_seed[i] = z ^ z >> 31;
  }
  return 0;
}
