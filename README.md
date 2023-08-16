Design of OpenMP-based Dynamic [RAK] algorithm for [community detection].

This is an implementation of a popular label-propagation based community
detection algorithm called **Raghavan Albert Kumara (RAK)**. Here, every node is
initialized with a unique label and at every step each node adopts the label
that most of its neighbors currently have. In this iterative process densely
connected groups of nodes form a consensus on a unique label to form
communities.

When there exist multiple communities with max weight, we randomly pick one of
them (**non-strict** approach), or pick only the first of them (**strict** approach).
The algorithm converges when `n%` of vertices dont change their community
membership (*tolerance*).

We continue with OpenMP implementation of RAK algorithm for community detection.
Each thread is given a *separate* hashtable, which it can use for choosing the
most popular label among its neighbors (by weight). I initially packed the
hashtables (for each thread) contiguously on a vector. However, i observed that
*allocating them separately* given almost *2x* performance (by using pointer to
vectors). OpenMP schedule is `auto` now, we can later choose the best if we
need.

There are three different dynamic approaches (as with [sequential]) i am trying out:
- **Naive-dynamic**: We simply use the previous community memberships and perform the algorithm.
- **Dynamic Delta-screening**: We find a set of affected vertices as per the [Delta-screening] paper.
- **Dyanmic Frontier**: We mark endpoints of each vertex as affected, and expand out iteratively.

The input data used for below experiments is available from the [SuiteSparse Matrix Collection].
The experiments were done with guidance from [Prof. Kishore Kothapalli] and
[Prof. Dip Sankar Banerjee].

[RAK]: https://arxiv.org/abs/0709.2938
[community detection]: https://en.wikipedia.org/wiki/Community_search
[sequential]: https://github.com/puzzlef/rak-communities-static-vs-dynamic
[Delta-screening]: https://ieeexplore.ieee.org/document/9384277
[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://faculty.iiit.ac.in/~kkishore/
[SuiteSparse Matrix Collection]: https://sparse.tamu.edu

<br>


### Comparision on large graphs

In this experiment ([input-large]), we first compute the community membership of
each vertex using the static RAK algorithm. We then generate random batch
updates consisting of an equal mix of *deletions (-)* and  *insertions (+)* of
edges of size `10^-7 |E|` to `0.1 |E|` in multiples of `10` (where `|E|` is the
number of edges in the original graph after making it undirected). For each
batch size, we generate *five* different batches for the purpose of *averaging*.
Each batch of edges (insertion / deletion) is generated randomly such that the
selection of each vertex (as endpoint) is *equally probable*.

We make the following observations. Dynamic approaches are faster than static
approaches on average. Among the dynamic approaches, *Dynamic Frontier* approach
is the **fastest**, followed by *Naive-dynamic*, and finally *Dynamic*
*Delta-screening* on average.

> See
> [code](https://github.com/puzzlef/rak-communities-openmp-dynamic/tree/input-large),
> [output](https://gist.github.com/wolfram77/ad7dd582d6e57c22c29ee4f24bc82797), or
> [sheets].

[![](https://i.imgur.com/68Ox0yW.png)][sheets]
[![](https://i.imgur.com/pyL1sZa.png)][sheets]
[![](https://i.imgur.com/44xQ8vp.png)][sheets]
[![](https://i.imgur.com/3snKKhY.png)][sheets]
[![](https://i.imgur.com/qA0QZVl.png)][sheets]
[![](https://i.imgur.com/i2yXEbn.png)][sheets]

[input-large]: https://github.com/puzzlef/rak-communities-openmp-dynamic/tree/input-large
[sheets]: https://docs.google.com/spreadsheets/d/1MG1NlpQ-etbwaENJwSjcq5eUTvsUphi2QfvYbjKnFss/edit?usp=sharing

<br>


### Measure affected vertices

In this experiment ([measure-affected]), we **measure** the number of **affected**
**vertices** with *Dynamic Delta-screening* and *Dynamic Frontier* based *RAK*
for random batch updates consisting of edge insertions, with the size of batch
update varying from `10^-6 |E|` to `0.1 |E|`.

Results show that *Dynamic Delta-screening* marks `770x`, `210x`, `71x`, `19x`,
`4.6x`, and `1.7x` the number of affected vertices as *Dynamic Frontier* based
approach (*strict RAK*) on batch updates of size `10^-6 |E|` to `0.1 |E|`.

[measure-affected]: https://github.com/puzzlef/rak-communities-openmp-dynamic/tree/measure-affected

<br>


### Multi-batch updates

In this experiment ([multi-batch]), we generate `5000` random **multi-batch**
**updates** consisting of *edge insertions* of size `10^-3 |E|` one after the
other on graphs `web-Stanford` and `web-BerkStan` and observe the performance
and modularity of communities obtained with *Static*, *Naive-dynamic*, *Dynamic*
*Delta-screening*, and *Dynamic Frontier* based *strict RAK*. We do this to
measure after how many batch updates do we need to re-run the static algorithm.

Our results indicate that we need to do not need to rerun the static algorithm
with *Dynamic Delta-screening* based *RAK* or with *Dynamic Frontier* based
*RAK*.

[multi-batch]: https://github.com/puzzlef/rak-communities-openmp-dynamic/tree/multi-batch

<br>
<br>


## Build instructions

To run the [input-large] experiment, download this repository and run the
following. Note that input graphs must be placed in `~/Data` directory, and
output logs will be written to `~/Logs` directory.

```bash
# Perform comparision on large graphs
$ DOWNLOAD=0 ./mains.sh

# Perform comparision on large graphs with custom number of threads
$ DOWNLOAD=0 MAX_THREADS=4 ./mains.sh
```

<br>
<br>


## References

- [Near linear time algorithm to detect community structures in large-scale networks; Usha Nandini Raghavan et al. (2007)](https://arxiv.org/abs/0709.2938)
- [Delta-Screening: A Fast and Efficient Technique to Update Communities in Dynamic Graphs](https://ieeexplore.ieee.org/document/9384277)
- [The University of Florida Sparse Matrix Collection; Timothy A. Davis et al. (2011)](https://doi.org/10.1145/2049662.2049663)
- [How to import VSCode keybindings into Visual Studio?](https://stackoverflow.com/a/62417446/1413259)
- [Configure X11 Forwarding with PuTTY and Xming](https://www.centlinux.com/2019/01/configure-x11-forwarding-putty-xming-windows.html)
- [Installing snap on CentOS](https://snapcraft.io/docs/installing-snap-on-centos)

<br>
<br>


[![](https://i.imgur.com/u17N4wL.jpg)](https://www.youtube.com/watch?v=JZO-ZwkFoF8)<br>
[![ORG](https://img.shields.io/badge/org-puzzlef-green?logo=Org)](https://puzzlef.github.io)
