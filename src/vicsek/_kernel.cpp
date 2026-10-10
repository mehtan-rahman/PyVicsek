// C++ kernel for the neighbour phase of the Vicsek step.
//
// Two functions, both with exactly the semantics of the numpy cell list in
// vicsek/util/cell_list.py (same neighbour set, same pair order, same
// floating-point operations in the same order, so the results are bit for bit
// identical when the compiler does not contract a*b+c into fused
// multiply-adds; CMakeLists.txt passes -ffp-contract=off):
//
//   neighbor_pairs(positions, box_length, radius, use_pbc=True, n_threads=1)
//       -> (i, j, distance)
//   mean_neighbor_velocity(positions, velocities, box_length, radius,
//                          use_pbc=True, n_threads=1) -> (mean, n_neighbors)
//
// Algorithm: counting sort of the particles into n_cells^dim cells (n_cells =
// max(1, floor(L / r)) per axis, cell index truncated then reduced modulo
// n_cells under periodic boundaries or clipped otherwise), then for every
// focal particle a sweep over the distinct neighbouring cells of the 3^dim
// stencil with the minimum-image distance. Any dimension >= 1.
//
// The GIL is released for the whole computation. With n_threads > 1 the
// focal particles are split into contiguous blocks over std::thread; every
// particle's result is computed by one thread in the same order as with one
// thread, so the output does not depend on n_threads.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

namespace nb = nanobind;

using InArray = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using OutF64 = nb::ndarray<nb::numpy, double>;
using OutI64 = nb::ndarray<nb::numpy, int64_t>;

namespace {

struct CellGrid {
    int64_t n = 0;        // particles
    int dim = 0;
    double length = 0;
    double radius = 0;
    bool pbc = true;
    int64_t n_cells = 1;  // per axis
    double cell_size = 0;

    std::vector<int64_t> strides;     // n_cells^(dim-1-k): axis 0 most significant
    std::vector<int> offsets;         // (n_offsets, dim), lexicographic, last axis fastest
    int n_offsets = 0;
    std::vector<int64_t> coords;      // (n, dim) cell coordinates, -1 rows for excluded particles
    std::vector<int64_t> start;       // (n_cells^dim + 1) offsets into order
    std::vector<int64_t> order;       // particle ids sorted by flat cell, stable
    std::vector<double> sorted_pos;   // positions in `order`, (m, dim)
    std::vector<char> included;       // particle takes part (always 1 under PBC)
};

// Python's floor modulo for a non-negative modulus.
inline int64_t floor_mod(int64_t a, int64_t m) {
    int64_t r = a % m;
    return r < 0 ? r + m : r;
}

void build_grid(CellGrid &g, const double *pos, int64_t n, int dim, double length,
                double radius, bool pbc) {
    if (dim < 1)
        throw std::invalid_argument("positions must have at least one column");
    if (!(length > 0) || !std::isfinite(length))
        throw std::invalid_argument("box_length must be positive and finite");
    if (!(radius > 0) || !std::isfinite(radius))
        throw std::invalid_argument("radius must be positive and finite");

    g.n = n;
    g.dim = dim;
    g.length = length;
    g.radius = radius;
    g.pbc = pbc;

    // _grid(): n_cells = max(1, int(floor(L / r))), cell_size = L / n_cells
    double cells_per_axis = std::floor(length / radius);
    double total_cells = std::pow(std::max(1.0, cells_per_axis), dim);
    if (total_cells > 1e9)
        throw std::length_error("too many cells (box_length / radius too large for this dimension)");
    g.n_cells = std::max<int64_t>(1, static_cast<int64_t>(cells_per_axis));
    g.cell_size = length / static_cast<double>(g.n_cells);

    g.strides.assign(dim, 1);
    for (int k = dim - 2; k >= 0; --k)
        g.strides[k] = g.strides[k + 1] * g.n_cells;
    const int64_t n_flat = g.strides[0] * g.n_cells;

    // _cell_offsets(): {-1, 0, 1} per axis, reduced to the distinct values
    // modulo n_cells under PBC with fewer than three cells per axis.
    std::vector<int> axis;
    if (pbc && g.n_cells == 1)
        axis = {0};
    else if (pbc && g.n_cells == 2)
        axis = {0, 1};
    else
        axis = {-1, 0, 1};
    const int a = static_cast<int>(axis.size());
    g.n_offsets = 1;
    for (int k = 0; k < dim; ++k)
        g.n_offsets *= a;
    g.offsets.resize(static_cast<size_t>(g.n_offsets) * dim);
    for (int o = 0; o < g.n_offsets; ++o) {
        int rest = o;
        for (int k = dim - 1; k >= 0; --k) {
            g.offsets[static_cast<size_t>(o) * dim + k] = axis[rest % a];
            rest /= a;
        }
    }

    // _bin_particles(): truncate, then modulo (PBC) or clip (open box).
    // Without PBC only particles inside [0, L) on every axis take part.
    g.coords.assign(static_cast<size_t>(n) * dim, -1);
    g.included.assign(n, 1);
    std::vector<int64_t> flat(n, -1);
    std::vector<int64_t> counts(n_flat, 0);
    for (int64_t i = 0; i < n; ++i) {
        const double *p = pos + i * dim;
        bool inside = true;
        for (int k = 0; k < dim; ++k) {
            if (!std::isfinite(p[k]))
                throw std::invalid_argument("positions must be finite");
            if (!pbc && !(p[k] >= 0 && p[k] < length))
                inside = false;
        }
        if (!inside) {
            g.included[i] = 0;
            continue;
        }
        int64_t f = 0;
        for (int k = 0; k < dim; ++k) {
            double q = p[k] / g.cell_size;
            if (!(std::fabs(q) < 4e18))
                throw std::invalid_argument("positions too far outside the box");
            int64_t c = static_cast<int64_t>(q);   // truncation, as astype(int64)
            c = pbc ? floor_mod(c, g.n_cells) : std::clamp<int64_t>(c, 0, g.n_cells - 1);
            g.coords[static_cast<size_t>(i) * dim + k] = c;
            f += c * g.strides[k];
        }
        flat[i] = f;
        ++counts[f];
    }

    // Counting sort, stable in particle index (argsort(kind="stable")).
    g.start.assign(n_flat + 1, 0);
    for (int64_t c = 0; c < n_flat; ++c)
        g.start[c + 1] = g.start[c] + counts[c];
    const int64_t m = g.start[n_flat];
    g.order.resize(m);
    g.sorted_pos.resize(static_cast<size_t>(m) * dim);
    std::vector<int64_t> fill(g.start.begin(), g.start.end() - 1);
    for (int64_t i = 0; i < n; ++i) {
        if (flat[i] < 0)
            continue;
        int64_t s = fill[flat[i]]++;
        g.order[s] = i;
        std::copy(pos + i * dim, pos + (i + 1) * dim, g.sorted_pos.begin() + s * dim);
    }
}

// Visit every neighbour j != i of particle i within the radius, in the order
// of the numpy implementation (stencil offsets in order, each cell's
// particles by increasing index). visit(j, slot_in_sorted_order, distance).
// D is the dimension when known at compile time (2 or 3), 0 otherwise.
template <int D, typename Visit>
inline void for_each_neighbor(const CellGrid &g, const double *pos, int64_t i, Visit &&visit) {
    if (!g.included[i])
        return;
    const int dim = D > 0 ? D : g.dim;
    const double *pi = pos + i * dim;
    const int64_t *ci = g.coords.data() + static_cast<size_t>(i) * dim;
    const double L = g.length;
    const double half_box = 0.49 * L;
    const double reject_above = g.radius * g.radius * (1.0 + 1e-12);
    for (int o = 0; o < g.n_offsets; ++o) {
        const int *off = g.offsets.data() + static_cast<size_t>(o) * dim;
        int64_t f = 0;
        bool valid = true;
        for (int k = 0; k < dim; ++k) {
            int64_t c = ci[k] + off[k];
            if (g.pbc) {
                if (c < 0)
                    c += g.n_cells;
                else if (c >= g.n_cells)
                    c -= g.n_cells;
            } else if (c < 0 || c >= g.n_cells) {
                valid = false;
                break;
            }
            f += c * g.strides[k];
        }
        if (!valid)
            continue;
        const int64_t end = g.start[f + 1];
        for (int64_t s = g.start[f]; s < end; ++s) {
            const int64_t j = g.order[s];
            const double *pj = g.sorted_pos.data() + s * dim;
            double squared = 0.0;
            for (int k = 0; k < dim; ++k) {
                double delta = pj[k] - pi[k];
                // delta - L * rint(delta / L); for |delta| < 0.49 L the
                // quotient rounds to zero and delta is unchanged, so the
                // division is skipped without changing a bit of the result.
                if (g.pbc && !(std::fabs(delta) < half_box)) {
                    double image = L * std::nearbyint(delta / L);
                    delta = delta - image;
                }
                double sq = delta * delta;
                squared = (k == 0) ? sq : squared + sq;
            }
            // sqrt(squared) <= r is decided exactly below; this cheap test
            // only rejects pairs clearly out of range (margin 1e-12 relative).
            if (squared > reject_above || j == i)
                continue;
            const double distance = std::sqrt(squared);
            if (distance <= g.radius)
                visit(j, s, distance);
        }
    }
}

// Number of contiguous blocks of focal particles: at most n_threads, and at
// least 256 particles per block so tiny inputs do not pay for thread start-up.
int64_t blocks_for(int64_t n, int n_threads) {
    return std::max<int64_t>(1, std::min<int64_t>(n_threads, n / 256));
}

// body(block, lo, hi) for every block; block 0 runs on the calling thread.
template <typename Body>
void parallel_blocks(int64_t n, int n_threads, Body &&body) {
    const int64_t t = blocks_for(n, n_threads);
    if (t == 1) {
        body(0, 0, n);
        return;
    }
    std::vector<std::thread> workers;
    workers.reserve(t - 1);
    for (int64_t w = 1; w < t; ++w)
        workers.emplace_back([&body, w, n, t] { body(static_cast<int>(w), n * w / t, n * (w + 1) / t); });
    body(0, 0, n / t);
    for (auto &th : workers)
        th.join();
}

// f(std::integral_constant<int, D>) with D = dim for 2 and 3, else D = 0.
template <typename F>
void with_dim(int dim, F &&f) {
    if (dim == 2)
        f(std::integral_constant<int, 2>{});
    else if (dim == 3)
        f(std::integral_constant<int, 3>{});
    else
        f(std::integral_constant<int, 0>{});
}

template <typename T>
nb::capsule owner_of(T *data) {
    return nb::capsule(data, [](void *p) noexcept { delete[] static_cast<T *>(p); });
}

std::tuple<OutI64, OutI64, OutF64>
neighbor_pairs(InArray positions, double box_length, double radius, bool use_pbc, int n_threads) {
    const int64_t n = static_cast<int64_t>(positions.shape(0));
    const int dim = static_cast<int>(positions.shape(1));
    const double *pos = positions.data();
    if (n_threads < 1)
        throw std::invalid_argument("n_threads must be at least 1");

    const int64_t t = blocks_for(n, n_threads);
    std::vector<std::vector<int64_t>> fi(t), fj(t);
    std::vector<std::vector<double>> fd(t);
    {
        nb::gil_scoped_release release;
        CellGrid g;
        build_grid(g, pos, n, dim, box_length, radius, use_pbc);
        with_dim(dim, [&](auto d_const) {
            constexpr int D = decltype(d_const)::value;
            parallel_blocks(n, n_threads, [&](int w, int64_t lo, int64_t hi) {
                auto &vi = fi[w], &vj = fj[w];
                auto &vd = fd[w];
                for (int64_t i = lo; i < hi; ++i)
                    for_each_neighbor<D>(g, pos, i, [&](int64_t j, int64_t, double d) {
                        vi.push_back(i);
                        vj.push_back(j);
                        vd.push_back(d);
                    });
            });
        });
    }

    size_t total = 0;
    for (auto &v : fi)
        total += v.size();
    int64_t *out_i = new int64_t[std::max<size_t>(total, 1)];
    int64_t *out_j = new int64_t[std::max<size_t>(total, 1)];
    double *out_d = new double[std::max<size_t>(total, 1)];
    nb::capsule oi = owner_of(out_i), oj = owner_of(out_j), od = owner_of(out_d);
    size_t at = 0;
    for (int64_t w = 0; w < t; ++w) {
        std::copy(fi[w].begin(), fi[w].end(), out_i + at);
        std::copy(fj[w].begin(), fj[w].end(), out_j + at);
        std::copy(fd[w].begin(), fd[w].end(), out_d + at);
        at += fi[w].size();
    }
    return {OutI64(out_i, {total}, oi), OutI64(out_j, {total}, oj), OutF64(out_d, {total}, od)};
}

std::tuple<OutF64, OutI64>
mean_neighbor_velocity(InArray positions, InArray velocities, double box_length, double radius,
                       bool use_pbc, int n_threads) {
    const int64_t n = static_cast<int64_t>(positions.shape(0));
    const int dim = static_cast<int>(positions.shape(1));
    if (static_cast<int64_t>(velocities.shape(0)) != n || static_cast<int>(velocities.shape(1)) != dim)
        throw std::invalid_argument("positions and velocities must have the same shape");
    if (n_threads < 1)
        throw std::invalid_argument("n_threads must be at least 1");
    const double *pos = positions.data();
    const double *vel = velocities.data();

    double *mean = new double[std::max<int64_t>(n * dim, 1)];
    int64_t *count = new int64_t[std::max<int64_t>(n, 1)];
    nb::capsule om = owner_of(mean), oc = owner_of(count);
    {
        nb::gil_scoped_release release;
        CellGrid g;
        build_grid(g, pos, n, dim, box_length, radius, use_pbc);
        // Velocities in cell order, so a neighbour cell is read contiguously.
        std::vector<double> sorted_vel(g.order.size() * dim);
        for (size_t s = 0; s < g.order.size(); ++s)
            std::copy(vel + g.order[s] * dim, vel + (g.order[s] + 1) * dim, sorted_vel.begin() + s * dim);
        // Particles outside an open box have no neighbours: they keep their velocity.
        for (int64_t i = 0; i < n; ++i)
            if (!g.included[i]) {
                std::copy(vel + i * dim, vel + (i + 1) * dim, mean + i * dim);
                count[i] = 0;
            }
        const int64_t m = static_cast<int64_t>(g.order.size());
        with_dim(dim, [&](auto d_const) {
            constexpr int D = decltype(d_const)::value;
            // Focal particles in cell order too (better locality; each
            // particle's result does not depend on the visiting order).
            parallel_blocks(m, n_threads, [&](int, int64_t lo, int64_t hi) {
                std::vector<double> acc(dim);
                for (int64_t s = lo; s < hi; ++s) {
                    const int64_t i = g.order[s];
                    // bincount: a running sum from 0 over the neighbours in
                    // pair order, then added to the particle's own velocity.
                    std::fill(acc.begin(), acc.end(), 0.0);
                    int64_t c = 0;
                    for_each_neighbor<D>(g, pos, i, [&](int64_t, int64_t slot, double) {
                        const double *vj = sorted_vel.data() + slot * dim;
                        for (int k = 0; k < (D > 0 ? D : dim); ++k)
                            acc[k] += vj[k];
                        ++c;
                    });
                    const double denom = static_cast<double>(c + 1);
                    const double *vi = vel + i * dim;
                    for (int k = 0; k < dim; ++k)
                        mean[i * dim + k] = (vi[k] + acc[k]) / denom;
                    count[i] = c;
                }
            });
        });
    }
    return {OutF64(mean, {static_cast<size_t>(n), static_cast<size_t>(dim)}, om),
            OutI64(count, {static_cast<size_t>(n)}, oc)};
}

}  // namespace

NB_MODULE(_kernel, m) {
    m.doc() = "C++ kernel for the Vicsek neighbour phase (see vicsek.util.cell_list).";

    m.def("neighbor_pairs", &neighbor_pairs, nb::arg("positions"), nb::arg("box_length"),
          nb::arg("radius"), nb::arg("use_pbc") = true, nb::arg("n_threads") = 1,
          "All ordered neighbour pairs (i, j, distance) with i != j and distance <= radius,\n"
          "grouped by i; identical to vicsek.util.cell_list.neighbor_pairs.");

    m.def("mean_neighbor_velocity", &mean_neighbor_velocity, nb::arg("positions"),
          nb::arg("velocities"), nb::arg("box_length"), nb::arg("radius"),
          nb::arg("use_pbc") = true, nb::arg("n_threads") = 1,
          "(mean, n_neighbors): mean velocity over each particle and its neighbours, and the\n"
          "neighbour count excluding the particle; identical to CellList.mean_neighbor_velocity.");
}
