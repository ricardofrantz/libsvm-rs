# libsvm-rs Developer Documentation

Detailed internal documentation for the pure Rust reimplementation of LIBSVM.

## Architecture Overview

```
crates/libsvm/src/
  lib.rs       — Module declarations and re-exports
  types.rs     — SvmNode, SvmProblem, SvmParameter, SvmModel, enums
  error.rs     — SvmError enum (thiserror)
  io.rs        — Problem/model file I/O (LIBSVM text format)
  kernel.rs    — Kernel evaluation (dot, powi, k_function, Kernel struct)
  cache.rs     — LRU kernel cache (Qfloat = f32)
  qmatrix.rs   — QMatrix trait + SvcQ, OneClassQ, SvrQ implementations
  solver.rs    — SMO solver (Standard + Nu variants)
  train.rs     — svm_train, solve dispatchers, class grouping
  predict.rs   — predict, predict_values
```

## Module Details

### types.rs

Core data types matching LIBSVM's `svm.h`:

| Type | Description | C++ equivalent |
|------|-------------|----------------|
| `SvmType` | Enum: CSvc, NuSvc, OneClass, EpsilonSvr, NuSvr | `svm_type` int constants |
| `KernelType` | Enum: Linear, Polynomial, Rbf, Sigmoid, Precomputed | `kernel_type` int constants |
| `SvmNode` | Sparse feature `{index: i32, value: f64}` | `struct svm_node` |
| `SvmProblem` | Training data: `labels: Vec<f64>`, `instances: Vec<Vec<SvmNode>>` | `struct svm_problem` |
| `SvmParameter` | All training parameters (defaults match LIBSVM) | `struct svm_parameter` |
| `SvmModel` | Trained model with SVs, coefficients, rho | `struct svm_model` |

Key design choice: `SvmNode.index` is `i32` (not `usize`) to match LIBSVM's C `int` and preserve file format compatibility. Sentinel nodes (`index == -1`) are not stored — Rust `Vec::len()` tracks instance length instead.

### kernel.rs

Two kernel evaluation paths:

1. **`k_function(x, y, param)`** — Standalone evaluation for prediction. Operates on sparse node slices.
2. **`Kernel` struct** — Training evaluator with precomputed `x_square` for RBF.

The `Kernel` struct stores `Vec<&'a [SvmNode]>` (a vec of borrowed slices into the original data). This allows the solver to swap data point references during shrinking without owning the data. The C++ code achieves the same via `const_cast` on a `const svm_node **x` pointer array.

**RBF optimization**: For RBF kernels, `x_square[i] = dot(x[i], x[i])` is precomputed. The kernel value is then:
```
K(i,j) = exp(-γ * (x_sq[i] + x_sq[j] - 2·dot(x[i], x[j])))
```
This avoids recomputing ‖x_i‖² on every evaluation.

### cache.rs

LRU kernel cache storing rows of the Q matrix as `Qfloat` (`f32`) values.

**Data structure**: Index-based circular doubly-linked list (no `unsafe`). Node `l` is the sentinel head. Nodes not in the LRU list have `prev == NONE` (usize::MAX).

**`get_data(index, len)`**: Returns `(&mut [Qfloat], start)` where `start` is how many elements were already cached. The caller fills `data[start..len]` with kernel values.

**`swap_index(i, j)`**: Critical for solver shrinking. Three steps:
1. Swap row data and LRU positions for rows i and j
2. Iterate ALL cached rows and swap column entries at positions i,j
3. If a row covers the lower index but not the higher, evict it ("give up")

This column-swap loop (missing from early versions) is essential for correctness — without it, the solver reads stale kernel values after shrinking swaps.

### qmatrix.rs

The `QMatrix` trait bridges the Kernel and Solver:

```rust
pub trait QMatrix {
    fn get_q(&mut self, i: usize, len: usize) -> &[Qfloat];
    fn get_qd(&self) -> &[f64];
    fn swap_index(&mut self, i: usize, j: usize);
}
```

**`&mut self` on `get_q`**: The C++ uses `const` methods with `mutable` cache. In Rust, we take `&mut self`, which means the Solver owns `Box<dyn QMatrix>` and copies Q row data into its own buffers to avoid borrow conflicts.

#### SvcQ

For C-SVC and ν-SVC classification:
- `Q[i][j] = y[i] * y[j] * K(x[i], x[j])` stored as `Qfloat` (f32)
- `QD[i] = K(x[i], x[i])` (always 1.0 for RBF)
- `swap_index` swaps cache, kernel data, labels, and diagonal

#### OneClassQ

For one-class SVM:
- `Q[i][j] = K(x[i], x[j])` — no label scaling
- Same cache and swap logic as SvcQ, minus the label array

#### SvrQ

For ε-SVR and ν-SVR regression. The dual has 2l variables:
- Variables 0..l have `sign[k] = +1`, `index[k] = k`
- Variables l..2l have `sign[k+l] = -1`, `index[k+l] = k`

The kernel cache stores only l rows of actual kernel values. `get_q` fetches the real kernel row, then reorders and applies signs into a **double buffer**:
```
buf[j] = sign[i] * sign[j] * data[index[j]]
```
Two buffers are needed because the solver requests Q_i and Q_j simultaneously during the alpha update. `swap_index` only swaps `sign`, `index`, and `qd` — the kernel cache maps to real data indices via `index[]`.

### solver.rs

The SMO solver — the computational core. Translates `Solver` and `Solver_NU` from svm.cpp (lines 362–1265).

**Solver variant**: `SolverVariant::Standard` vs `SolverVariant::Nu`. 95% shared code — only 4 methods differ:
- `select_working_set`
- `calculate_rho`
- `do_shrinking`
- `be_shrunk`

**Key constants**:
| Constant | Value | Purpose |
|----------|-------|---------|
| `TAU` | 1e-12 | Floor for quad_coef to avoid division by zero |
| `INF` | f64::INFINITY | Initial gradient bounds |

#### Initialization

1. Copy input arrays (alpha, y, p) into owned vectors
2. Compute alpha_status for each variable (LowerBound/UpperBound/Free)
3. Initialize gradient: `G[i] = p[i] + Σ_j(alpha[j] * Q[i][j])` for non-lower-bound j
4. Initialize G_bar: `G_bar[i] = Σ_j(C_j * Q[i][j])` for upper-bound j

#### Main Loop

```
while iter < max_iter:
    if counter expires: do_shrinking()
    (i, j) = select_working_set()  // returns None if optimal
    if None:
        reconstruct_gradient()     // unshrink all variables
        active_size = l
        (i, j) = select_working_set()  // retry
        if None: break  // truly optimal
    update_alpha_pair(i, j)        // analytic 2-variable solve
```

**max_iter**: `max(10_000_000, 100*l)` — same as C++.
**shrink_counter**: `min(l, 1000)` iterations between shrinking passes.

#### Working Set Selection (WSS3)

**Standard variant**: Two-pass heuristic:
1. Find i: maximizes `-y_i * G[i]` among I_up (can increase alpha)
2. Find j: minimizes objective decrease `-(grad_diff²) / quad_coef` among I_low

**Nu variant**: Separates positive and negative classes. Finds `i_p` (best positive) and `i_n` (best negative), then selects j from the matching class.

#### Alpha Update

Analytic solution of the 2-variable sub-problem with box constraints:
- **Different labels** (y[i] ≠ y[j]): constraint `alpha[i] - alpha[j] = constant`
- **Same labels**: constraint `alpha[i] + alpha[j] = constant`

Both branches clip alpha to [0, C] box and update gradients:
```
G[k] += Q_i[k] * Δα_i + Q_j[k] * Δα_j
```

G_bar is updated if alpha_status changes (transition to/from upper bound).

#### Shrinking

Variables at their bounds that are unlikely to change get swapped to the end of the active set:
- `be_shrunk(i)`: checks if `-y_i * G[i]` exceeds the current violation bound
- Active-from-back swap: finds the first non-shrinkable variable from the end

**Unshrink trigger**: When `Gmax1 + Gmax2 <= eps * 10`, reconstructs the full gradient and resets active_size = l.

#### Rho Calculation

**Standard**: `rho = average(y_i * G_i)` over free variables, or midpoint of bounds if no free variables.

**Nu**: Separate rho for positive (r1) and negative (r2) classes:
- Returns `rho = (r1 - r2) / 2`
- Stores `r = (r1 + r2) / 2` for subsequent alpha rescaling

### train.rs

The training pipeline orchestrating the solver.

#### Solve Dispatchers

| Function | SVM Type | Solver | Key Setup |
|----------|----------|--------|-----------|
| `solve_c_svc` | C-SVC | Standard | p = [-1,...], alpha *= y[i] after |
| `solve_nu_svc` | ν-SVC | Nu | alpha init from nu budget, rescale by 1/r |
| `solve_one_class` | One-class | Standard | alpha = [1,1,...,frac,0,...] from nu*l |
| `solve_epsilon_svr` | ε-SVR | Standard | 2l variables, p = ε ± y_i |
| `solve_nu_svr` | ν-SVR | Nu | 2l variables, alpha init from C·ν·l/2 |

#### Class Grouping

`svm_group_classes` builds:
- `label[]`: unique class labels (in first-occurrence order, with -1/+1 swap)
- `count[]`: samples per class
- `start[]`: cumulative start index per class
- `perm[]`: permutation mapping grouped indices to original indices

#### svm_train

Two branches:
1. **Regression/one-class**: Single call to `svm_train_one`, extract nonzero alphas as SVs
2. **Classification**: One-vs-one loop:
   - Group classes
   - Compute weighted C per class
   - For each pair (i,j): build sub-problem, train, mark nonzero
   - Assemble sv_coef matrix: `sv_coef[j-1][nz_start[i]..] = class_i coefficients`

**Gamma auto-detection**: If `gamma == 0`, sets `gamma = 1/max_feature_index`.

### predict.rs

Prediction for all SVM types:
- **Classification** (C-SVC, ν-SVC): One-vs-one voting over k*(k-1)/2 pairwise decision values
- **Regression** (ε-SVR, ν-SVR): `Σ sv_coef[i] * K(x, sv[i]) - rho`
- **One-class**: Sign of decision value

### io.rs

LIBSVM-compatible text format I/O:
- **Problem files**: `label index:value index:value ...` per line
- **Model files**: Header section + SV section. Float formatting uses `%.17g`-equivalent (matches C's printf)

The `Gfmt` struct replicates C's `%g` format: picks fixed vs scientific notation, strips trailing zeros.

## Numerical Equivalence Notes

- **Gradient accumulation**: Uses f64 for gradients (matching C++), f32 for cached Q values
- **Quad_coef floor**: TAU = 1e-12 prevents division by zero
- **Alpha comparison**: `alpha[i].abs() > 0.0` for SV detection (exact zero check, matching C++)
- **Class label cast**: `labels[i] as i32` matches C's `(int)prob->y[i]` truncation

## Test Coverage

| Category | Tests | Description |
|----------|-------|-------------|
| Cache | 7 | LRU eviction, extend, swap with column updates |
| Kernel | 8 | All kernel types, struct/standalone agreement |
| QMatrix | 4 | SvcQ sign/symmetry, OneClassQ, SvrQ double buffer |
| I/O | 8 | Problem parsing, model roundtrip, C compatibility |
| Types | 8 | Parameter validation, ν-SVC feasibility |
| Predict | 3 | Heart_scale accuracy, C svm-predict comparison |
| Train | 6 | C-SVC, multiclass, ν-SVC, one-class, ε-SVR, ν-SVR |
| **Total** | **50** | |

## Dependencies

- **Runtime**: `thiserror` only (zero other deps)
- **Optional**: `rayon` (feature-gated, for future parallel CV)
- **Dev**: `float-cmp`, `criterion`, `proptest`
