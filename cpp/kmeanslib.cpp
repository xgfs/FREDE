#include <math.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace std;

#ifdef _WIN32
#define DLLEXPORT extern "C" __declspec(dllexport)
#else
#define DLLEXPORT extern "C"
#endif

typedef unsigned long long ull;
#ifndef INT64_C
#define INT64_C(c) (c##LL)
#define UINT64_C(c) (c##ULL)
#endif

#define DEFAULT_ALIGN 128

uint64_t rng_seed[2];

static uint64_t rotl(const uint64_t x, int k) {
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

static int irand(int min, int max) { return lrand() % (max - min) + min; }
static int irand(int max) { return lrand() % max; }

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

DLLEXPORT double modularity(int *assignment, int k, const int *offsets,
                            const int *edges, const int nv) {
  auto inc = static_cast<double *>(calloc(k, sizeof(double)));
  auto deg = static_cast<double *>(calloc(k, sizeof(double)));
  for (auto i = 0; i < nv; i++) {
    auto com = assignment[i];
    deg[com] += offsets[i + 1] - offsets[i];
    for (auto adj = offsets[i]; adj < offsets[i + 1]; adj++) {
      if (assignment[edges[adj]] == com) inc[com] += 0.5f;
    }
  }
  int ne = offsets[nv];
  double result = 0;
  for (auto i = 0; i < k; i++)
    result += 2.0 * inc[i] / ne - pow(static_cast<double>(deg[i]) / ne, 2);
  free(inc);
  free(deg);
  return result;
}

void init_random(float *mu, float *w, const int k, const int nv,
                 const int dim) {
  auto inc = static_cast<int32_t *>(malloc(k * sizeof(int32_t)));
  for (auto i = 0; i < k; i++) {
    bool keep = true;
    do {
      inc[i] = irand(nv);
      for (auto j = 0; j < i; j++)
        if (inc[i] == inc[j]) keep = false;
    } while (!keep);
    memcpy(&mu[i * dim], &w[inc[i] * dim], dim * sizeof(float));
  }
  free(inc);
}

int discrete_rand(float *array, int sz) {
  double sum = 0;
#pragma omp simd
  for (auto i = 0; i < sz; i++) sum += array[i];
  auto r = drand() * sum;
  double cumsum = array[0];
  int newk = 0;
  while (r >= cumsum && newk < sz - 1) {
    newk++;
    cumsum += array[newk];
  }
  return newk;
}

void init_plusplus(float *mu, float *w, const int k, const int nv,
                   const int dim) {
  auto minDist = static_cast<float *>(malloc(nv * sizeof(float)));
  auto curDist = static_cast<float *>(malloc(nv * sizeof(float)));
  memset(minDist, 0, nv * sizeof(float));
  auto m0 = irand(nv);
  memcpy(mu, &w[m0 * dim], dim * sizeof(float));
  for (auto i = 1; i < k; i++) {
    for (auto j = 0; j < nv; j++)
#pragma simd
      for (auto kk = 0; kk < dim; kk++)
        curDist[j] += pow(mu[(i - 1) * dim + kk] - w[j * dim + kk], 2);
    if (i == 1)
      memcpy(minDist, curDist, nv * sizeof(float));
    else
#pragma simd
      for (auto j = 0; j < nv; j++) minDist[j] = min(minDist[j], curDist[j]);
    auto mi = discrete_rand(minDist, nv);
    memcpy(&mu[i * dim], &w[mi * dim], dim * sizeof(float));
  }
  free(minDist);
  free(curDist);
}

void pairwise_distances(float *mu, float *w, float *dist, int kk, const int nv,
                        const int dim, int n_threads = 1) {
  memset(dist, 0, nv * kk * sizeof(float));
#pragma omp parallel for num_threads(n_threads)
  for (auto i = 0; i < nv; i++)
    for (auto j = 0; j < kk; j++)
#pragma omp simd
      for (auto k = 0; k < dim; k++)
        dist[i * kk + j] += pow(mu[j * dim + k] - w[i * dim + k], 2);
}

double assign_closest(float *mu, float *w, float *dist, int *z, int kk,
                      const int nv, const int dim, int n_threads = 1) {
  double totalDist = 0;
  pairwise_distances(mu, w, dist, kk, nv, dim, n_threads);
#pragma omp parallel for num_threads(n_threads)
  for (int i = 0; i < nv; i++) {
    int minRowID = 0;
    double minRowVal = dist[i * kk];
    for (int j = 1; j < kk; j++)
      if (dist[i * kk + j] < minRowVal) {
        minRowID = j;
        minRowVal = dist[i * kk + j];
      }
    z[i] = minRowID;
#pragma omp atomic
    totalDist += minRowVal;
  }
  return totalDist;
}

void update_mu(float *mu, float *w, int *z, int kk, const int nv,
               const int dim) {
  memset(mu, 0, dim * kk * sizeof(float));
  auto inc = static_cast<int32_t *>(calloc(kk, sizeof(int32_t)));
  for (int i = 0; i < nv; i++) {
    int cls = z[i];
    inc[cls]++;
#pragma simd
    for (int j = 0; j < dim; j++) mu[cls * dim + j] += w[i * dim + j];
  }
  for (int i = 0; i < kk; i++)
    if (inc[i] == 0) inc[i]++;
  for (int i = 0; i < kk; i++)
#pragma simd
    for (int j = 0; j < dim; j++) mu[i * dim + j] /= inc[i];
  free(inc);
}

DLLEXPORT double run_kmeans(int *z, float *w, int k, const int nv,
                            const int dim, int niter = 100, int n_threads = 1) {
  auto mu = static_cast<float *>(malloc(k * dim * sizeof(float)));
  auto dist = static_cast<float *>(malloc(k * nv * sizeof(float)));
  memset(z, 0, nv * sizeof(int));
  init_plusplus(mu, w, k, nv, dim);
  // init_random(mu, w, k, nv, dim);
  double prevDist = 9999999999999;
  for (int i = 0; i < niter; i++) {
    auto totalDist = assign_closest(mu, w, dist, z, k, nv, dim, n_threads);
    update_mu(mu, w, z, k, nv, dim);
    if (prevDist <= totalDist) break;
    prevDist = totalDist;
  }
  free(dist);
  free(mu);
  return prevDist;
}