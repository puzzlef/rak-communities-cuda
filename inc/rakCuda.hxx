#pragma once
#include <tuple>
#include <vector>
#include <algorithm>
#include <cstdint>
#include "_main.hxx"
#include "rak.hxx"
#include "hashtableCuda.hxx"

using std::tuple;
using std::vector;
using std::count_if;
using std::partition;




#pragma region LAUNCH CONFIG
#ifndef BLOCK_LIMIT_RAK_THREAD_CUDA
/** Maximum number of threads per block with RAK thread-per-vertex kernel. */
#define BLOCK_LIMIT_RAK_THREAD_CUDA  32
/** Maximum number of threads per block with RAK block-per-vertex kernel. */
#define BLOCK_LIMIT_RAK_BLOCK_CUDA   128
#endif
#pragma endregion




#pragma region METHODS
#pragma region INITIALIZE
/**
 * Initialize communities such that each vertex is its own community [kernel].
 * @param vcom community each vertex belongs to (output)
 * @param NB begin vertes (inclusive)
 * @param NE end vertex (exclusive)
 */
template <class K>
void __global__ rakInitializeCukW(K *vcom, K NB, K NE) {
  DEFINE_CUDA(t, b, B, G);
  for (K u=NB+B*b+t; u<NE; u+=G*B)
    vcom[u] = u;
}


/**
 * Initialize communities such that each vertex is its own community.
 * @param vcom community each vertex belongs to (output)
 * @param NB begin vertes (inclusive)
 * @param NE end vertex (exclusive)
 */
template <class K>
inline void rakInitializeCuW(K *vcom, K NB, K NE) {
  const int B = blockSizeCu(NE-NB,   BLOCK_LIMIT_MAP_CUDA);
  const int G = gridSizeCu (NE-NB, B, GRID_LIMIT_MAP_CUDA);
  rakInitializeCukW<<<G, B>>>(vcom, NB, NE);
}
#pragma endregion




#pragma region UPDATE COMMUNITY MEMBERSHIPS
/**
 * Scan communities connected to a vertex [device function].
 * @tparam SELF include self-loops?
 * @tparam BLOCK called from a thread block?
 * @param hk hashtable keys (updated)
 * @param hv hashtable values (updated)
 * @param H capacity of hashtable (prime)
 * @param T secondary prime (>H)
 * @param xoff offsets of original graph
 * @param xedg edge keys of original graph
 * @param xwei edge values of original graph
 * @param u given vertex
 * @param vcom community each vertex belongs to
 * @param i start index
 * @param DI index stride
 */
template <bool SELF=false, bool BLOCK=false, class O, class K, class V, class W>
inline void __device__ rakScanCommunitiesCudU(K *hk, W *hv, size_t H, size_t T, const O *xoff, const K *xedg, const V *xwei, K u, const K *vcom, size_t i, size_t DI) {
  size_t EO = xoff[u];
  size_t EN = xoff[u+1] - xoff[u];
  for (; i<EN; i+=DI) {
    K v = xedg[EO+i];
    W w = xwei[EO+i];
    K c = vcom[v];
    if (!SELF && u==v) continue;
    hashtableAccumulateCudU<BLOCK>(hk, hv, H, T, c+1, w);
  }
}


/**
 * Mark out-neighbors of a vertex as affected [device function].
 * @param vaff vertex affected flags (updated)
 * @param xoff offsets of original graph
 * @param xedg edge keys of original graph
 * @param u given vertex
 * @param i start index
 * @param DI index stride
 */
template <class O, class K, class F>
inline void __device__ rakMarkNeighborsCudU(F *vaff, const O *xoff, const K *xedg, K u, size_t i, size_t DI) {
  size_t EO = xoff[u];
  size_t EN = xoff[u+1] - xoff[u];
  for (; i<EN; i+=DI) {
    K v = xedg[EO+i];
    vaff[v] = F(1);  // TODO: Use two (synchronous) buffers?
  }
}


/**
 * Move each vertex to its best community, using thread-per-vertex approach [kernel].
 * @param ncom number of changed vertices (updated)
 * @param vcom community each vertex belongs to (updated)
 * @param vaff vertex affected flags (updated)
 * @param bufk buffer for hashtable keys (updated)
 * @param bufw buffer for hashtable values (updated)
 * @param xoff offsets of original graph
 * @param xedg edge keys of original graph
 * @param xwei edge values of original graph
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 * @param PICKLESS allow only picking smaller community id?
 */
template <class O, class K, class V, class W, class F>
void __global__ rakMoveIterationThreadCukU(uint64_cu *ncom, K *vcom, F *vaff, K *bufk, W *bufw, const O *xoff, const K *xedg, const V *xwei, K NB, K NE, bool PICKLESS) {
  DEFINE_CUDA(t, b, B, G);
  __shared__ uint64_cu ncomb[BLOCK_LIMIT_RAK_THREAD_CUDA];
  const int  MAX_DEGREE = BLOCK_LIMIT_RAK_THREAD_CUDA;
  K shrk[2 * MAX_DEGREE];
  W shrw[2 * MAX_DEGREE];
  ncomb[t] = 0;
  for (K u=NB+B*b+t; u<NE; u+=G*B) {
    if (!vaff[u]) continue;
    // Scan communities connected to u.
    K d = vcom[u];
    // size_t EO = xoff[u];
    size_t EN = xoff[u+1] - xoff[u];
    size_t H  = nextPow2Cud(EN) - 1;
    size_t T  = nextPow2Cud(H)  - 1;
    K *hk = shrk;  // bufk + 2*EO;
    W *hv = shrw;  // bufw + 2*EO;
    hashtableClearCudW(hk, hv, H, 0, 1);
    rakScanCommunitiesCudU(hk, hv, H, T, xoff, xedg, xwei, u, vcom, 0, 1);
    // Find best community for u.
    hashtableMaxCudU(hk, hv, H, 0, 1);
    vaff[u] = F(0);         // Mark u as unaffected (TODO: Use two buffers?)
    if  (!hk[0]) continue;  // No community found
    K c = hk[0] - 1;        // Best community
    if (c==d) continue;
    if (PICKLESS && c>d) continue;  // Pick smaller community-id (to avoid community swaps)
    // Change community of u.
    vcom[u] = c; ++ncomb[t];
    rakMarkNeighborsCudU(vaff, xoff, xedg, u, 0, 1);
  }
  // Update number of changed vertices.
  __syncthreads();
  sumValuesBlockReduceCudU(ncomb, B, t);
  if (t==0) atomicAdd(ncom, ncomb[0]);
}


/**
 * Move each vertex to its best community, using thread-per-vertex approach.
 * @param ncom number of changed vertices (updated)
 * @param vcom community each vertex belongs to (updated)
 * @param vaff vertex affected flags (updated)
 * @param bufk buffer for hashtable keys (updated)
 * @param bufw buffer for hashtable values (updated)
 * @param xoff offsets of original graph
 * @param xedg edge keys of original graph
 * @param xwei edge values of original graph
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 * @param PICKLESS allow only picking smaller community id?
 */
template <class O, class K, class V, class W, class F>
inline void rakMoveIterationThreadCuU(uint64_cu *ncom, K *vcom, F *vaff, K *bufk, W *bufw, const O *xoff, const K *xedg, const V *xwei, K NB, K NE, bool PICKLESS) {
  const int B = blockSizeCu(NE-NB,   BLOCK_LIMIT_RAK_THREAD_CUDA);
  const int G = gridSizeCu (NE-NB, B, GRID_LIMIT_MAP_CUDA);
  rakMoveIterationThreadCukU<<<G, B>>>(ncom, vcom, vaff, bufk, bufw, xoff, xedg, xwei, NB, NE, PICKLESS);
}


/**
 * Move each vertex to its best community, using block-per-vertex approach [kernel].
 * @param ncom number of changed vertices (updated)
 * @param vcom community each vertex belongs to (updated)
 * @param vaff vertex affected flags (updated)
 * @param bufk buffer for hashtable keys (updated)
 * @param bufw buffer for hashtable values (updated)
 * @param xoff offsets of original graph
 * @param xedg edge keys of original graph
 * @param xwei edge values of original graph
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 * @param PICKLESS allow only picking smaller community id?
 */
template <class O, class K, class V, class W, class F>
void __global__ rakMoveIterationBlockCukU(uint64_cu *ncom, K *vcom, F *vaff, K *bufk, W *bufw, const O *xoff, const K *xedg, const V *xwei, K NB, K NE, bool PICKLESS) {
  DEFINE_CUDA(t, b, B, G);
  uint64_cu ncomb = 0;
  for (K u=NB+b; u<NE; u+=G) {
    if (!vaff[u]) continue;
    // Scan communities connected to u.
    K d = vcom[u];
    size_t EO = xoff[u];
    size_t EN = xoff[u+1] - xoff[u];
    size_t H  = nextPow2Cud(EN) - 1;
    size_t T  = nextPow2Cud(H)  - 1;
    K *hk = bufk + 2*EO;
    W *hv = bufw + 2*EO;
    hashtableClearCudW(hk, hv, H, t, B);
    __syncthreads();
    rakScanCommunitiesCudU<false, true>(hk, hv, H, T, xoff, xedg, xwei, u, vcom, t, B);
    __syncthreads();
    // Find best community for u.
    hashtableMaxCudU<true>(hk, hv, H, t, B);
    __syncthreads();
    if (t==0) vaff[u] = F(0);  // Mark u as unaffected (TODO: Use two buffers?)
    if  (!hk[0]) continue;     // No community found
    K c = hk[0] - 1;           // Best community
    if (c==d) continue;
    if (PICKLESS && c>d) continue;  // Pick smaller community-id (to avoid community swaps)
    // Change community of u.
    if (t==0) vcom[u] = c;
    if (t==0) ++ncomb;
    rakMarkNeighborsCudU(vaff, xoff, xedg, u, t, B);
  }
  // Update number of changed vertices.
  if (t==0) atomicAdd((uint64_cu*) ncom, ncomb);
}


/**
 * Move each vertex to its best community, using block-per-vertex approach.
 * @param ncom number of changed vertices (output)
 * @param vcom community each vertex belongs to (updated)
 * @param vaff vertex affected flags (updated)
 * @param bufk buffer for hashtable keys (updated)
 * @param bufw buffer for hashtable values (updated)
 * @param xoff offsets of original graph
 * @param xedg edge keys of original graph
 * @param xwei edge values of original graph
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 * @param PICKLESS allow only picking smaller community id?
 */
template <class O, class K, class V, class W, class F>
inline void rakMoveIterationBlockCuU(uint64_cu *ncom, K *vcom, F *vaff, K *bufk, W *bufw, const O *xoff, const K *xedg, const V *xwei, K NB, K NE, bool PICKLESS) {
  const int B = blockSizeCu<true>(NE-NB,   BLOCK_LIMIT_RAK_BLOCK_CUDA);
  const int G = gridSizeCu <true>(NE-NB, B, GRID_LIMIT_MAP_CUDA);
  rakMoveIterationBlockCukU<<<G, B>>>(ncom, vcom, vaff, bufk, bufw, xoff, xedg, xwei, NB, NE, PICKLESS);
}
#pragma endregion




#pragma region CROSS CHECK
template <class K>
void __global__ rakCrossCheckCukU(uint64_cu *ncom, K *vcom, K *vdom, K NB, K NE) {
  DEFINE_CUDA(t, b, B, G);
  for (K u=NB+B*b+t; u<NE; u+=G*B) {
    K c = vdom[u];
    if (vdom[u]==vcom[u]) continue;
    if (vdom[c]==c) vcom[u] = c;
    else atomicAdd(ncom, uint64_cu()-1);  // Bad things happened
  }
}


template <class K>
inline void rakCrossCheckCuU(uint64_cu *ncom, K *vcom, K *vdom, K NB, K NE) {
  const int B = blockSizeCu(NE-NB,   BLOCK_LIMIT_MAP_CUDA);
  const int G = gridSizeCu (NE-NB, B, GRID_LIMIT_MAP_CUDA);
  rakCrossCheckCukU<<<G, B>>>(ncom, vcom, vdom, NB, NE);
}
#pragma endregion




#pragma region COMPUTATION LOOP
/**
 * Perform RAK iterations.
 * @tparam PICKSTEP allow only picking smaller community id, every PICKSTEP iterations?
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
template <class O, class K, class V, class W, class F>
inline int rakLoopCuU(uint64_cu *ncom, K *vcom, F *vaff, K *bufk, W *bufw, const O *xoff, const K *xedg, const V *xwei, K N, K NL, double E, int L) {
  int l = 0;
  uint64_cu n = 0;
  const int PICKSTEP = 4;
  while (l<L) {
    bool PICKLESS = l % PICKSTEP == 0;
    fillValueCuW(ncom, 1, uint64_cu());
    rakMoveIterationThreadCuU(ncom, vcom, vaff, bufk, bufw, xoff, xedg, xwei, K(), NL, PICKLESS);
    rakMoveIterationBlockCuU (ncom, vcom, vaff, bufk, bufw, xoff, xedg, xwei, NL,  N,  PICKLESS); ++l;
    // rakCrossCheckCuU(ncom, vdom, vcom, K(), N); swap(vdom, vcom);
    TRY_CUDA( cudaMemcpy(&n, ncom, sizeof(uint64_cu), cudaMemcpyDeviceToHost) );
    if (!PICKLESS && double(n)/N <= E) break;
  }
  return l;
}
#pragma endregion




#pragma region PARTITION
/**
 * Partition vertices into low-degree and high-degree sets.
 * @param ks vertex keys (updated)
 * @param x original graph
 * @returns number of low-degree vertices
 */
template <class G, class K>
inline size_t rakPartitionVerticesCudaU(vector<K>& ks, const G& x) {
  K SWITCH_DEGREE = 32;  // Switch to block-per-vertex approach if degree >= SWITCH_DEGREE
  K SWITCH_LIMIT  = 64;  // Avoid switching if number of vertices < SWITCH_LIMIT
  size_t N = ks.size();
  auto  kb = ks.begin(), ke = ks.end();
  auto  ft = [&](K v) { return x.degree(v) < SWITCH_DEGREE; };
  partition(kb, ke, ft);
  size_t n = count_if(kb, ke, ft);
  if (n   < SWITCH_LIMIT) n = 0;
  if (N-n < SWITCH_LIMIT) n = N;
  return n;
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
template <class FLAG=char, class G, class K, class FM>
inline RakResult<K> rakInvokeCuda(const G& x, const vector<K>* q, const RakOptions& o, FM fm) {
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
    // Mark initial affected vertices.
    tm += mark([&]() { fm(vaff); });
    gatherValuesW(vaffc, vaff, ks);
    TRY_CUDA( cudaMemcpy(vaffD, vaffc.data(), N * sizeof(F), cudaMemcpyHostToDevice) );
    // Perform RAK iterations.
    mark([&]() { l = rakLoopCuU(ncomD, vcomD, vaffD, bufkD, bufwD, xoffD, xedgD, xweiD, K(N), K(NL), E, L); });
  }, o.repeat);
  // Obtain final community membership.
  TRY_CUDA( cudaMemcpy(vcomc.data(), vcomD, N * sizeof(K), cudaMemcpyDeviceToHost) );
  scatterValuesW(vcom, vcomc, ks);
  // Free device memory.
  TRY_CUDA( cudaFree(xoffD) );
  TRY_CUDA( cudaFree(xedgD) );
  TRY_CUDA( cudaFree(xweiD) );
  TRY_CUDA( cudaFree(vcomD) );
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
template <class FLAG=char, class G, class K>
inline RakResult<K> rakStaticCuda(const G& x, const vector<K>* q=nullptr, const RakOptions& o={}) {
  auto fm = [](auto& vaff) { fillValueOmpU(vaff, FLAG(1)); };
  return rakInvokeCuda<FLAG>(x, q, o, fm);
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
template <class FLAG=char, class G, class K, class V>
inline RakResult<K> rakDynamicFrontierCuda(const G& y, const vector<tuple<K, K>>& deletions, const vector<tuple<K, K, V>>& insertions, const vector<K>* q, const RakOptions& o={}) {
  auto fm = [&](auto& vaff) { rakAffectedVerticesFrontierOmpW(vaff, y, deletions, insertions, *q); };
  return rakInvokeCuda<FLAG>(y, q, o, fm);
}
#pragma endregion
#pragma endregion
