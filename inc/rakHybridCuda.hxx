#pragma once
#include <tuple>
#include <vector>
#include <algorithm>
#include <cstdint>
#include "_main.hxx"
#include "rak.hxx"
#include "rakCuda.hxx"
#include "rakPicklessCuda.hxx"

using std::tuple;
using std::vector;
using std::count_if;
using std::partition;




#pragma region METHODS
#pragma region COMPUTATION LOOP
/**
 * Perform RAK iterations.
 * @param ncom number of changed vertices (updated)
 * @param vcom community each vertex belongs to (updated)
 * @param vaff vertex affected flags (updated)
 * @param bufk buffer for hashtable keys (updated)
 * @param bufw buffer for hashtable values (updated)
 * @param xoff offsets of original graph
 * @param xedg edge keys of original graph
 * @param xwei edge values of original graph
 * @param N number of vertices
 * @param NL number of low-degree vertices
 * @param E tolerance for convergence [0.05]
 * @param L maximum number of iterations [20]
 * @returns number of iterations performed
 */
template <int CHECKSTEP=1, int PICKSTEP=1, class O, class K, class V, class W, class F>
inline int rakHybridLoopCuU(uint64_cu *ncom, K *&vcom, K *&vdom, F *vaff, K *bufk, W *bufw, const O *xoff, const K *xedg, const V *xwei, K N, K NL, double E, int L) {
  int l = 0;
  uint64_cu n = 0;
  while (l<L) {
    bool CROSSCHECK = l % CHECKSTEP == 0;
    bool PICKLESS   = l % PICKSTEP  == 0;
    fillValueCuW(ncom, 1, uint64_cu());
    rakPicklessMoveIterationThreadCuU(ncom, vcom, vaff, bufk, bufw, xoff, xedg, xwei, K(), NL, PICKLESS);
    rakPicklessMoveIterationBlockCuU (ncom, vcom, vaff, bufk, bufw, xoff, xedg, xwei, NL,  N,  PICKLESS); ++l;
    if (CROSSCHECK) { rakCrossCheckCuU(ncom, vdom, vcom, K(), N); swap(vdom, vcom); }
    TRY_CUDA( cudaMemcpy(&n, ncom, sizeof(uint64_cu), cudaMemcpyDeviceToHost) );
    if ((PICKSTEP==1 || !PICKLESS) && double(n)/N <= E) break;
  }
  return l;
}
#pragma endregion




#pragma region ENVIRONMENT SETUP
/**
 * Setup and perform the RAK algorithm.
 * @tparam FLAG flag type for tracking affected vertices
 * @param x original graph
 * @param q initial community each vertex belongs to
 * @param o rak options
 * @param fm marking affected vertices / preprocessing to be performed (vaff)
 * @returns rak result
 */
template <int CHECKSTEP=1, int PICKSTEP=1, class FLAG=char, class G, class K, class FM>
inline RakResult<K> rakHybridInvokeCuda(const G& x, const vector<K>* q, const RakOptions& o, FM fm) {
  using V = typename G::edge_value_type;
  using W = RAK_WEIGHT_TYPE;
  using O = uint32_t;
  using F = FLAG;
  size_t S = x.span();
  size_t N = x.order();
  size_t M = x.size();
  int    R = reduceSizeCu(N);
  int    L = o.maxIterations, l = 0;
  double E = o.tolerance;
  vector<O> xoff(N+1);
  vector<K> xedg(M);
  vector<V> xwei(M);
  vector<K> vcom(S), vcomc(N);
  vector<F> vaff(S), vaffc(N);
  O *xoffD = nullptr;
  K *xedgD = nullptr;
  V *xweiD = nullptr;
  K *vcomD = nullptr;
  K *vdomD = nullptr;
  F *vaffD = nullptr;
  K *bufkD = nullptr;
  W *bufwD = nullptr;
  uint64_cu *ncomD = nullptr;
  // Partition vertices into low-degree and high-degree sets.
  vector<K> ks = vertexKeys(x);
  size_t NL = rakPartitionVerticesCudaU(ks, x);
  // Obtain data for CSR.
  csrCreateOffsetsW (xoff, x, ks);
  csrCreateEdgeKeysW(xedg, x, ks);
  csrCreateEdgeValuesW(xwei, x, ks);
  // Obtain initial community membership.
  if (q) gatherValuesW(vcomc, *q, ks);
  // Allocate device memory.
  TRY_CUDA( cudaSetDeviceFlags(cudaDeviceMapHost) );
  TRY_CUDA( cudaMalloc(&xoffD, (N+1) * sizeof(O)) );
  TRY_CUDA( cudaMalloc(&xedgD,  M    * sizeof(K)) );
  TRY_CUDA( cudaMalloc(&xweiD,  M    * sizeof(V)) );
  TRY_CUDA( cudaMalloc(&vcomD,  N    * sizeof(K)) );
  TRY_CUDA( cudaMalloc(&vdomD,  N    * sizeof(K)) );
  TRY_CUDA( cudaMalloc(&vaffD,  N    * sizeof(F)) );
  TRY_CUDA( cudaMalloc(&bufkD, (2*M) * sizeof(K)) );
  TRY_CUDA( cudaMalloc(&bufwD, (2*M) * sizeof(W)) );
  TRY_CUDA( cudaMalloc(&ncomD,  1    * sizeof(uint64_cu)) );
  // Copy data to device.
  TRY_CUDA( cudaMemcpy(xoffD, xoff.data(), (N+1) * sizeof(O), cudaMemcpyHostToDevice) );
  TRY_CUDA( cudaMemcpy(xedgD, xedg.data(),  M    * sizeof(K), cudaMemcpyHostToDevice) );
  TRY_CUDA( cudaMemcpy(xweiD, xwei.data(),  M    * sizeof(V), cudaMemcpyHostToDevice) );
  // Perform RAK algorithm on device.
  float tm = 0;
  float t  = measureDurationMarked([&](auto mark) {
    // Setup initial community membership.
    if (q) TRY_CUDA( cudaMemcpy(vcomD, vcomc.data(), N * sizeof(K), cudaMemcpyHostToDevice) );
    else   rakInitializeCuW(vcomD, K(), K(N));
    if (q) TRY_CUDA( cudaMemcpy(vdomD, vcomc.data(), N * sizeof(K), cudaMemcpyHostToDevice) );
    else   rakInitializeCuW(vdomD, K(), K(N));
    // Mark initial affected vertices.
    tm += mark([&]() { fm(vaff); });
    gatherValuesW(vaffc, vaff, ks);
    TRY_CUDA( cudaMemcpy(vaffD, vaffc.data(), N * sizeof(F), cudaMemcpyHostToDevice) );
    // Perform RAK iterations.
    mark([&]() { l = rakHybridLoopCuU<CHECKSTEP, PICKSTEP>(ncomD, vcomD, vdomD, vaffD, bufkD, bufwD, xoffD, xedgD, xweiD, K(N), K(NL), E, L); });
  }, o.repeat);
  // Obtain final community membership.
  TRY_CUDA( cudaMemcpy(vcomc.data(), vcomD, N * sizeof(K), cudaMemcpyDeviceToHost) );
  scatterValuesW(vcom, vcomc, ks);
  // Free device memory.
  TRY_CUDA( cudaFree(xoffD) );
  TRY_CUDA( cudaFree(xedgD) );
  TRY_CUDA( cudaFree(xweiD) );
  TRY_CUDA( cudaFree(vcomD) );
  TRY_CUDA( cudaFree(vdomD) );
  TRY_CUDA( cudaFree(vaffD) );
  TRY_CUDA( cudaFree(bufkD) );
  TRY_CUDA( cudaFree(bufwD) );
  TRY_CUDA( cudaFree(ncomD) );
  return {vcom, l, t, tm/o.repeat};
}
#pragma endregion




#pragma region STATIC/NAIVE-DYNAMIC
/**
 * Obtain the community membership of each vertex with Static/Naive-dynamic RAK.
 * @tparam FLAG flag type for tracking affected vertices
 * @param x original graph
 * @param q initial community each vertex belongs to
 * @param o rak options
 * @returns rak result
 */
template <int CHECKSTEP=1, int PICKSTEP=1, class FLAG=char, class G, class K>
inline RakResult<K> rakHybridStaticCuda(const G& x, const vector<K>* q=nullptr, const RakOptions& o={}) {
  auto fm = [](auto& vaff) { fillValueOmpU(vaff, FLAG(1)); };
  return rakHybridInvokeCuda<CHECKSTEP, PICKSTEP, FLAG>(x, q, o, fm);
}
#pragma endregion




#pragma region DYNAMIC FRONTIER APPROACH
/**
 * Obtain the community membership of each vertex with Dynamic Frontier RAK.
 * @tparam FLAG flag type for tracking affected vertices
 * @param y updated graph
 * @param deletions edge deletions in batch update
 * @param insertions edge insertions in batch update
 * @param q initial community each vertex belongs to
 * @param o rak options
 * @returns rak result
 */
template <int CHECKSTEP=1, int PICKSTEP=1, class FLAG=char, class G, class K, class V>
inline RakResult<K> rakHybridDynamicFrontierCuda(const G& y, const vector<tuple<K, K>>& deletions, const vector<tuple<K, K, V>>& insertions, const vector<K>* q, const RakOptions& o={}) {
  auto fm = [&](auto& vaff) { rakAffectedVerticesFrontierOmpW(vaff, y, deletions, insertions, *q); };
  return rakHybridInvokeCuda<CHECKSTEP, PICKSTEP, FLAG>(y, q, o, fm);
}
#pragma endregion
#pragma endregion
