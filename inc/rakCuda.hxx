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
#pragma endregion
