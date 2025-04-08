Design of CUDA-based [Label Propagation Algorithm (LPA)], aka RAK, for [community detection].

Community detection is the problem of identifying natural divisions in networks. Efficient parallel algorithms for identifying such divisions are critical in a number of applications. This report presents an optimized implementation of the Label Propagation Algorithm (LPA) for community detection, featuring an asynchronous LPA with a Pick-Less (PL) method every `4` iterations to handle community swaps, ideal for SIMT hardware like GPUs. It also introduces a novel per-vertex hashtable with hybrid quadratic-double probing for collision resolution. On an NVIDIA A100 GPU, our implementation, ν-LPA, outperforms FLPA, NetworKit LPA, and GVE-LPA by `364x`, `62x`, and `2.6x`, respectively, on a server with dual 16-core Intel Xeon Gold 6226R processors - processing `3.0B` edges/s on a `2.2B` edge graph - and achieves `4.7%` higher modularity than FLPA, but `6.1%` and `2.2%` lower than NetworKit LPA and GVE-LPA.

Below we plot the time taken by [FLPA], [NetworKit] LPA, [GVE-LPA], and ν-LPA on 13 different graphs. ν-LPA surpasses FLPA, NetworKit LPA, and GVE-LPA by `364`, `62×`, and `2.6×` respectively, achieving a processing rate of `3.0B` edges/s on a `2.2𝐵` edge graph.

[![](https://i.imgur.com/EmuwFSf.png)][sheets-o1]

Below we plot the speedup of ν-LPA wrt FLPA, NetworKit LPA, and GVE-LPA.

[![](https://i.imgur.com/dAgIdcx.png)][sheets-o1]

Next, we plot the modularity of communities identified by FLPA, NetworKit LPA, GVE-LPA, and ν-LPA. ν-LPA on average obtains `4.7%`higher modularity than FLPA, but `6.1%` / `2.2%` lower modularity than NetworKit LPA / GVE-LPA. We recommend employing 𝜈-LPA on web graphs and social networks. For road networks, however, GVE-LPA appears to be the most effective, while NetworKit LPA is recommended for protein k-mer graphs.

[![](https://i.imgur.com/dqgA3ws.png)][sheets-o1]

Refer to our technical report for more details: \
[ν-LPA: Fast GPU-based Label Propagation Algorithm (LPA) for Community Detection][report].

<br>

> [!NOTE]
> You can just copy `main.sh` to your system and run it. \
> For the code, refer to `main.cxx`.


[Label Propagation Algorithm (LPA)]: https://arxiv.org/abs/0709.2938
[FLPA]: https://github.com/vtraag/igraph/tree/flpa
[NetworKit]: https://github.com/networkit/networkit
[GVE-LPA]: https://github.com/puzzlef/rak-communities-openmp
[community detection]: https://en.wikipedia.org/wiki/Community_search
[sheets-o1]: https://docs.google.com/spreadsheets/d/1cyfBYXXUT6NpjY2M1wkVBqExHTapM6Cx7agdJS2dS6s/edit?usp=sharing
[report]: https://arxiv.org/abs/2411.11468

<br>
<br>


### Code structure

The code structure of ν-LPA is as follows:

```bash
- inc/_algorithm.hxx: Algorithm utility functions
- inc/_bitset.hxx: Bitset manipulation functions
- inc/_cmath.hxx: Math functions
- inc/_ctypes.hxx: Data type utility functions
- inc/_cuda.hxx: CUDA utility functions
- inc/_debug.hxx: Debugging macros (LOG, ASSERT, ...)
- inc/_iostream.hxx: Input/output stream functions
- inc/_iterator.hxx: Iterator utility functions
- inc/_main.hxx: Main program header
- inc/_mpi.hxx: MPI (Message Passing Interface) utility functions
- inc/_openmp.hxx: OpenMP utility functions
- inc/_queue.hxx: Queue utility functions
- inc/_random.hxx: Random number generation functions
- inc/_string.hxx: String utility functions
- inc/_utility.hxx: Runtime measurement functions
- inc/_vector.hxx: Vector utility functions
- inc/batch.hxx: Batch update generation functions
- inc/bfs.hxx: Breadth-first search algorithms
- inc/csr.hxx: Compressed Sparse Row (CSR) data structure functions
- inc/dfs.hxx: Depth-first search algorithms
- inc/duplicate.hxx: Graph duplicating functions
- inc/Graph.hxx: Graph data structure functions
- inc/rak.hxx: LPA/RAK community detection algorithm functions
- inc/rakCuda.hxx: CUDA implementation of LPA (ν-LPA)
- inc/hashtableCuda.hxx: Open addressing hashtable functions, with quadratic-double probing
- inc/split.hxx: Algorithms to split internally-disconnected communities
- inc/main.hxx: Main header
- inc/mtx.hxx: Graph file reading functions
- inc/properties.hxx: Graph Property functions
- inc/selfLoop.hxx: Graph Self-looping functions
- inc/symmetricize.hxx: Graph Symmetricization functions
- inc/transpose.hxx: Graph transpose functions
- inc/update.hxx: Update functions
- main.cxx: Experimentation code
- process.js: Node.js script for processing output logs
```

Note that each branch in this repository contains code for a specific experiment. The `main` branch contains code for the final experiment. If the intention of a branch in unclear, or if you have comments on our technical report, feel free to open an issue.

<br>
<br>


## References

- [Near linear time algorithm to detect community structures in large-scale networks; Raghavan et al. (2007)](https://arxiv.org/abs/0709.2938)
- [The University of Florida Sparse Matrix Collection; Davis et al. (2011)](https://doi.org/10.1145/2049662.2049663)
- [How to import VSCode keybindings into Visual Studio?](https://stackoverflow.com/a/62417446/1413259)
- [Configure X11 Forwarding with PuTTY and Xming](https://www.centlinux.com/2019/01/configure-x11-forwarding-putty-xming-windows.html)
- [Installing snap on CentOS](https://snapcraft.io/docs/installing-snap-on-centos)

<br>
<br>


[![](https://i.imgur.com/7QLfaW3.jpg)](https://www.youtube.com/watch?v=IwiYQILYXDQ)<br>
[![ORG](https://img.shields.io/badge/org-puzzlef-green?logo=Org)](https://puzzlef.github.io)
![](https://ga-beacon.deno.dev/G-KD28SG54JQ:hbAybl6nQFOtmVxW4if3xw/github.com/puzzlef/rak-communities-cuda)

[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://faculty.iiit.ac.in/~kkishore/
[SuiteSparse Matrix Collection]: https://sparse.tamu.edu
