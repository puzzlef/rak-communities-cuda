#pragma once
#include <vector>
#include <cstdint>
#include "_main.hxx"
#include "rak.hxx"
#include "hashtableCuda.hxx"

using std::vector;




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
  K d = vcom[u];
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
 */
template <class O, class K, class V, class W, class F>
void __global__ rakMoveIterationThreadCukU(uint64_cu *ncom, K *vcom, F *vaff, K *bufk, W *bufw, const O *xoff, const K *xedg, const V *xwei, K NB, K NE) {
  DEFINE_CUDA(t, b, B, G);
  uint64_cu ncomt = 0;
  for (K u=NB+B*b+t; u<NE; u+=G*B) {
    if (!vaff[u]) continue;
    // Scan communities connected to u.
    K d = vcom[u];
    size_t EO = xoff[u];
    size_t EN = xoff[u+1] - xoff[u];
    size_t H  = nextPow2Cud(EN) - 1;
    size_t T  = nextPow2Cud(H)  - 1;
    K *hk = bufk + 2*EO;
    W *hv = bufw + 2*EO;
    hashtableClearCudW(hk, hv, H, 0, 1);
    rakScanCommunitiesCudU(hk, hv, H, T, xoff, xedg, xwei, u, vcom, 0, 1);
    // Find best community for u.
    hashtableMaxCudU(hk, hv, H, 0, 1);
    vaff[u] = F(0);         // Mark u as unaffected (TODO: Use two buffers?)
    if  (!hk[0]) continue;  // No community found
    K c = hk[0] - 1;        // Best community
    if (c==d) continue;
    // Change community of u.
    vcom[u] = c; ++ncomt;
    rakMarkNeighborsCudU(vaff, xoff, xedg, u, 0, 1);
  }
  atomicAdd(ncom, ncomt);
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
 */
template <class O, class K, class V, class W, class F>
inline void rakMoveIterationThreadCuU(uint64_t *ncom, K *vcom, F *vaff, K *bufk, W *bufw, const O *xoff, const K *xedg, const V *xwei, K NB, K NE) {
  const int B = blockSizeCu(NE-NB,   BLOCK_LIMIT_MAP_CUDA);
  const int G = gridSizeCu (NE-NB, B, GRID_LIMIT_MAP_CUDA);
  rakMoveIterationThreadCukU<<<G, B>>>(ncom, vcom, vaff, bufk, bufw, xoff, xedg, xwei, NB, NE);
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
 */
template <class O, class K, class V, class W, class F>
void __global__ rakMoveIterationBlockCukU(uint64_cu *ncom, K *vcom, F *vaff, K *bufk, W *bufw, const O *xoff, const K *xedg, const V *xwei, K NB, K NE) {
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
    // Change community of u.
    if (t==0) vcom[u] = c;
    if (t==0) ++ncomb;
    rakMarkNeighborsCudU(vaff, xoff, xedg, u, t, B);
  }
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
 */
template <class O, class K, class V, class W, class F>
inline void rakMoveIterationBlockCuU(uint64_cu *ncom, K *vcom, F *vaff, K *bufk, W *bufw, const O *xoff, const K *xedg, const V *xwei, K NB, K NE) {
  const int B = blockSizeCu<true>(NE-NB,   BLOCK_LIMIT_MAP_CUDA);
  const int G = gridSizeCu <true>(NE-NB, B, GRID_LIMIT_MAP_CUDA);
  rakMoveIterationBlockCukU<<<G, B>>>(ncom, vcom, vaff, bufk, bufw, xoff, xedg, xwei, NB, NE);
}
#pragma endregion




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
template <class O, class K, class V, class W, class F>
inline int rakLoopCuU(uint64_cu *ncom, K *vcom, F *vaff, K *bufk, W *bufw, const O *xoff, const K *xedg, const V *xwei, K N, K NL, double E, int L) {
  int l = 0;
  uint64_cu n = 0;
  while (l<L) {
    fillValueCuW(ncom, 1, uint64_cu());
    rakMoveIterationThreadCuU(ncom, vcom, vaff, bufk, bufw, xoff, xedg, xwei, K(), NL);
    rakMoveIterationBlockCuU (ncom, vcom, vaff, bufk, bufw, xoff, xedg, xwei, NL,  N); ++l;
    TRY_CUDA( cudaMemcpy(&n, ncom, sizeof(uint64_cu), cudaMemcpyDeviceToHost) );
    if (double(n)/N <= E) break;
  }
  return l;
}
#pragma endregion
#pragma endregion
