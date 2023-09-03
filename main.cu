#include <cstdint>
#include <cstdio>
#include <utility>
#include <random>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include "inc/main.hxx"

using namespace std;




#pragma region CONFIGURATION
#ifndef TYPE
/** Type of edge weights. */
#define TYPE float
#endif
#ifndef MAX_THREADS
/** Maximum number of threads to use. */
#define MAX_THREADS 64
#endif
#ifndef REPEAT_BATCH
/** Number of times to repeat each batch. */
#define REPEAT_BATCH 5
#endif
#ifndef REPEAT_METHOD
/** Number of times to repeat each method. */
#define REPEAT_METHOD 1
#endif
#pragma endregion




#pragma region METHODS
#pragma region HELPERS
/**
 * Obtain the modularity of community structure on a graph.
 * @param x original graph
 * @param a rak result
 * @param M sum of edge weights
 * @returns modularity
 */
template <class G, class K>
inline double getModularity(const G& x, const RakResult<K>& a, double M) {
  auto fc = [&](auto u) { return a.membership[u]; };
  return modularityBy(x, fc, M, 1.0);
}
#pragma endregion




#pragma region EXPERIMENTAL SETUP
/**
 * Run a function on each batch update, with a specified range of batch sizes.
 * @param x original graph
 * @param rnd random number generator
 * @param fn function to run on each batch update
 */
template <class G, class R, class F>
inline void runBatches(const G& x, R& rnd, F fn) {
  using  E = typename G::edge_value_type;
  double d = BATCH_DELETIONS_BEGIN;
  double i = BATCH_INSERTIONS_BEGIN;
  for (int epoch=0;; ++epoch) {
    for (int r=0; r<REPEAT_BATCH; ++r) {
      auto y  = duplicate(x);
      for (int sequence=0; sequence<BATCH_LENGTH; ++sequence) {
      auto deletions  = removeRandomEdges(y, rnd, size_t(d * x.size()/2), 1, x.span()-1);
      auto insertions = addRandomEdges   (y, rnd, size_t(i * x.size()/2), 1, x.span()-1, E(1));
        fn(y, d, deletions, i, insertions, sequence, epoch);
      }
    }
    if (d>=BATCH_DELETIONS_END && i>=BATCH_INSERTIONS_END) break;
    d BATCH_DELETIONS_STEP;
    i BATCH_INSERTIONS_STEP;
    d = min(d, double(BATCH_DELETIONS_END));
    i = min(i, double(BATCH_INSERTIONS_END));
  }
}


/**
 * Run a function on each number of threads, with a specified range of thread counts.
 * @param fn function to run on each number of threads
 */
template <class F>
inline void runThreads(F fn) {
  for (int t=NUM_THREADS_BEGIN; t<=NUM_THREADS_END; t NUM_THREADS_STEP) {
    omp_set_num_threads(t);
    fn(t);
    omp_set_num_threads(MAX_THREADS);
  }
}
#pragma endregion




#pragma region PERFORM EXPERIMENT
/**
 * Perform the experiment.
 * @param x original graph
 */
template <class G>
void runExperiment(const G& x) {
  using K = typename G::key_type;
  using V = typename G::edge_value_type;
  random_device dev;
  default_random_engine rnd(dev());
  int repeat  = REPEAT_METHOD;
  vector<K> *init = nullptr;
  double M = edgeWeightOmp(x)/2;
  // Follow a specific result logging format, which can be easily parsed later.
  auto glog = [&](const auto& ans, const char *technique, int numThreads, const auto& y, auto M, auto deletionsf, auto insertionsf) {
    printf(
      "{-%.3e/+%.3e batchf, %03d threads} -> "
      "{%09.1fms, %09.1fms preproc, %04d iters, %01.9f modularity} %s\n",
      double(deletionsf), double(insertionsf), numThreads,
      ans.time, ans.preprocessingTime, ans.iterations, getModularity(y, ans, M), technique
    );
  };
  // Get community memberships on original graph (static).
  auto d0 = rakStaticOmp(x, init, {5});
  glog(d0, "rakStaticOmpOriginal", MAX_THREADS, x, M, 0.0, 0.0);
  // Get community memberships on updated graph (dynamic).
  runBatches(x, rnd, [&](const auto& y, auto deletionsf, const auto& deletions, auto insertionsf, const auto& insertions, int sequence, int epoch) {
    double M = edgeWeightOmp(y)/2;
    // Adjust number of threads.
    runThreads([&](int numThreads) {
      auto flog = [&](const auto& ans, const char *technique) {
        glog(ans, technique, numThreads, y, M, deletionsf, insertionsf);
      };
      // Find static RAK (strict).
      auto d1 = rakStaticOmp(y, init, {repeat});
      flog(d1, "rakStaticOmp");
      {
        auto c1 = rakStaticCuda<1>(y, init, {repeat});
        flog(c1, "rakStaticCuda1");
        auto c2 = rakStaticCuda<2>(y, init, {repeat});
        flog(c2, "rakStaticCuda2");
        auto c3 = rakStaticCuda<3>(y, init, {repeat});
        flog(c3, "rakStaticCuda3");
        auto c4 = rakStaticCuda<4>(y, init, {repeat});
        flog(c4, "rakStaticCuda4");
      }
      {
        auto b1 = rakPicklessStaticCuda<1>(y, init, {repeat});
        flog(b1, "rakPicklessStaticCuda1");
        auto b2 = rakPicklessStaticCuda<2>(y, init, {repeat});
        flog(b2, "rakPicklessStaticCuda2");
        auto b3 = rakPicklessStaticCuda<3>(y, init, {repeat});
        flog(b3, "rakPicklessStaticCuda3");
        auto b4 = rakPicklessStaticCuda<4>(y, init, {repeat});
        flog(b4, "rakPicklessStaticCuda4");
      }
      {
        auto e1 = rakHybridStaticCuda<1, 1>(y, init, {repeat});
        flog(e1, "rakHybridStaticCuda11");
        auto e2 = rakHybridStaticCuda<1, 2>(y, init, {repeat});
        flog(e2, "rakHybridStaticCuda12");
        auto e3 = rakHybridStaticCuda<1, 3>(y, init, {repeat});
        flog(e3, "rakHybridStaticCuda13");
        auto e4 = rakHybridStaticCuda<1, 4>(y, init, {repeat});
        flog(e4, "rakHybridStaticCuda14");
        auto e5 = rakHybridStaticCuda<2, 1>(y, init, {repeat});
        flog(e5, "rakHybridStaticCuda21");
        auto e6 = rakHybridStaticCuda<2, 2>(y, init, {repeat});
        flog(e6, "rakHybridStaticCuda22");
        auto e7 = rakHybridStaticCuda<2, 3>(y, init, {repeat});
        flog(e7, "rakHybridStaticCuda23");
        auto e8 = rakHybridStaticCuda<2, 4>(y, init, {repeat});
        flog(e8, "rakHybridStaticCuda24");
        auto e9 = rakHybridStaticCuda<3, 1>(y, init, {repeat});
        flog(e9, "rakHybridStaticCuda31");
        auto e0 = rakHybridStaticCuda<3, 2>(y, init, {repeat});
        flog(e0, "rakHybridStaticCuda32");
        auto eA = rakHybridStaticCuda<3, 3>(y, init, {repeat});
        flog(eA, "rakHybridStaticCuda33");
        auto eB = rakHybridStaticCuda<3, 4>(y, init, {repeat});
        flog(eB, "rakHybridStaticCuda34");
        auto eC = rakHybridStaticCuda<4, 1>(y, init, {repeat});
        flog(eC, "rakHybridStaticCuda41");
        auto eD = rakHybridStaticCuda<4, 2>(y, init, {repeat});
        flog(eD, "rakHybridStaticCuda42");
        auto eE = rakHybridStaticCuda<4, 3>(y, init, {repeat});
        flog(eE, "rakHybridStaticCuda43");
        auto eF = rakHybridStaticCuda<4, 4>(y, init, {repeat});
        flog(eF, "rakHybridStaticCuda44");
      }
      // Find naive-dynamic RAK (strict).
      // auto d2 = rakStaticOmp(y, &d0.membership, {repeat});
      // flog(d2, "rakNaiveDynamicOmp");
      // auto c2 = rakStaticCuda(y, &d0.membership, {repeat});
      // flog(c2, "rakNaiveDynamicCuda");
      // auto b2 = rakPicklessStaticCuda(y, &d0.membership, {repeat});
      // flog(b2, "rakPicklessNaiveDynamicCuda");
      // Find frontier based dynamic RAK (strict).
      // auto d4 = rakDynamicFrontierOmp(y, deletions, insertions, &d0.membership, {repeat});
      // flog(d4, "rakDynamicFrontierOmp");
      // auto c4 = rakDynamicFrontierCuda(y, deletions, insertions, &d0.membership, {repeat});
      // flog(c4, "rakDynamicFrontierCuda");
      // auto b4 = rakPicklessDynamicFrontierCuda(y, deletions, insertions, &d0.membership, {repeat});
      // flog(b4, "rakPicklessDynamicFrontierCuda");
    });
  });
}


/**
 * Main function.
 * @param argc argument count
 * @param argv argument values
 * @returns zero on success, non-zero on failure
 */
int main(int argc, char **argv) {
  using K = uint32_t;
  using V = TYPE;
  install_sigsegv();
  char *file     = argv[1];
  bool symmetric = argc>2? stoi(argv[2]) : false;
  bool weighted  = argc>3? stoi(argv[3]) : false;
  omp_set_num_threads(MAX_THREADS);
  LOG("OMP_NUM_THREADS=%d\n", MAX_THREADS);
  LOG("Loading graph %s ...\n", file);
  DiGraph<K, None, V> x;
  readMtxOmpW(x, file, weighted); LOG(""); println(x);
  if (!symmetric) { x = symmetricizeOmp(x); LOG(""); print(x); printf(" (symmetricize)\n"); }
  runExperiment(x);
  printf("\n");
  return 0;
}
#pragma endregion
#pragma endregion
