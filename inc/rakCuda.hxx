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
#pragma endregion
#pragma endregion
