# AeroFlux

A zero-dependency, zero-heap, deterministic 3D incompressible Navier-Stokes
solver in Go, with a hardware-accelerated data plane on Apple Silicon via the
Accelerate framework.

The entire fluid state lives outside the Go heap in an `mmap` arena. The hot
path allocates nothing. Results are bit-for-bit identical regardless of how many
cores participate.

```go
e, err := aeroflux.New(aeroflux.Config{
    NX: 64, NY: 64, NZ: 64,
    H:         1.0 / 64,
    DT:        0.004,
    Viscosity: 1e-5,
})
if err != nil {
    return err
}
defer e.Close()

e.EnableSparseSolver() // Darwin: direct Cholesky solve via Accelerate

for i := 0; i < 1000; i++ {
    if err := e.Step(); err != nil {
        return err // a contained guard-page fault, not a crash
    }
}
fmt.Println(e.MaxDivergence()) // 0 to ~1e-14 with the direct solver
```

Go 1.27+. No third-party dependencies — `go.mod` has no `require` block.
Accelerate needs macOS with `CGO_ENABLED=1`; everything else runs on pure-Go
fallbacks.

---

# Part I — The Physics

## 1. The equations being solved

AeroFlux integrates the incompressible Navier-Stokes equations for a Newtonian
fluid of constant density:

```
∂u/∂t + (u·∇)u = −(1/ρ)∇p + ν∇²u        (momentum)
∇·u = 0                                   (continuity / incompressibility)
```

where `u = (u,v,w)` is the velocity field, `p` the pressure, `ρ` the density,
and `ν` the kinematic viscosity.

Read the momentum equation as Newton's second law per unit volume. The left
side is acceleration following a fluid parcel — the material derivative
`Du/Dt`. The right side is the forces: a pressure gradient pushing fluid from
high to low pressure, and viscous friction smoothing velocity differences.

The second equation is the one that makes this hard. It is **not** an evolution
equation — there is no `∂/∂t` in it. It is a *constraint* that must hold at
every instant. And notice what is missing: there is no equation of state, no
`∂p/∂t`. Pressure is not a thermodynamic variable being advanced in time here.

### What pressure actually is in this formulation

Pressure is a **Lagrange multiplier**. It is whatever scalar field is required,
at this instant, to keep the velocity divergence-free. In an incompressible
fluid, pressure information propagates at infinite speed: squeeze the fluid
anywhere and every other point must respond immediately, because no compression
is allowed to absorb the change. That is why the pressure solve is a *global*
elliptic problem rather than a local update — and why it dominates the runtime.

This is also why the code solves for pressure afresh every step rather than
storing and advancing it.

## 2. Operator splitting

Solving all the coupled terms simultaneously is expensive. AeroFlux instead uses
**fractional-step splitting** (Chorin's projection method), handling one physical
process at a time:

```
1. Advection    u* ← transport u by itself
2. Diffusion    u** ← apply viscous smoothing to u*
3. Projection   u^{n+1} ← remove the divergent part of u**
```

Each stage is individually tractable. The price is splitting error, which
Section 6 addresses.

### 2.1 Advection — semi-Lagrangian backtracing

The advection term `(u·∇)u` is nonlinear and is where turbulence comes from.
Rather than discretize it directly, AeroFlux uses the **semi-Lagrangian** method
(Stam's *Stable Fluids*), which exploits a physical fact: along a fluid parcel's
own path, transported quantities are constant.

So instead of asking "how does the value at this grid point change?", ask
"**where did the stuff now at this grid point come from?**" Trace backwards:

```
x_departure = x_arrival − Δt · u(x_arrival)
u_new(x_arrival) = interpolate(u_old, x_departure)
```

In [`stencil.go`](stencil.go):

```go
px -= dt * vx * invH              // backtrace (grid units)
adv := e.sampleTrilinear(src, sd, px, py, pz)
```

**Why this scheme:** it is *unconditionally stable*. An explicit upwind
discretization requires the CFL condition `|u|Δt/h ≤ 1` — the timestep is capped
by the fastest flow anywhere in the domain. Semi-Lagrangian has no such cap,
because it never extrapolates: it only reads values that already exist. A
departure point ten cells away is just as valid as one a tenth of a cell away.

**The cost:** trilinear interpolation averages eight neighbors, and repeated
averaging is a low-pass filter. This is *numerical dissipation* — energy leaks
out of the simulation not through physical viscosity but through interpolation
error. Vortices decay faster than they physically should. This is the
well-known weakness of the method, and Section 6's palindrome exists partly to
mitigate it.

**Departure point clamping.** `sampleTrilinear` clamps the departure coordinate
into the padded domain. This is a deliberate exception to the branchless-hot-loop
rule, and the reasoning is in the code: a trace landing just outside the halo
during ordinary near-boundary flow is routine, not catastrophic, so faulting on
it would kill a step for nothing. The guard pages remain the backstop for
genuine blow-ups — a NaN coordinate fails every comparison, so the clamp cannot
mask it, and it propagates to the index and faults.

### 2.2 Diffusion — explicit second differences

The viscous term `ν∇²u` is discretized with the standard second-order central
difference. Because the axes are swept separately (Section 6), each pass carries
only its own directional second difference:

```
∂²u/∂x² ≈ (u[i−1] − 2u[i] + u[i+1]) / h²
```

```go
lam = nu * dt / (h*h)
lap = lam * (sd[im] - 2*sd[i] + sd[ip])
```

Summed over the three axis passes, this reconstructs the full 7-point Laplacian.

**Stability.** Explicit diffusion is *conditionally* stable. Von Neumann analysis
gives, in 3D, `Δt ≤ h²/(6ν)`. Exceeding it makes the highest-frequency mode grow
each step and the simulation explodes within a handful of iterations.
`Config.Normalize` **rejects** such configurations rather than clamping them:

```go
limit := c.H * c.H / (6 * c.Viscosity)
if c.DT > limit {
    return fmt.Errorf("dt %g exceeds explicit diffusion stability limit %g ...")
}
```

A silently-adjusted timestep would produce plausible-looking but wrong physics —
strictly worse than an error.

Note the sharp `h²` scaling: halving the grid spacing quarters the maximum
stable timestep. This is why high-viscosity fine-grid runs want an implicit
diffusion solve, which AeroFlux does not currently implement.

### 2.3 Projection — Helmholtz-Hodge decomposition

After advection and diffusion, `u**` generally violates `∇·u = 0`. Fixing it
rests on the **Helmholtz-Hodge decomposition**: any sufficiently smooth vector
field on a domain with suitable boundary conditions splits *uniquely* into a
divergence-free part and a gradient:

```
u** = u_divergence-free + ∇φ
```

These two components are orthogonal in the L² inner product. So to extract the
divergence-free part, compute `∇φ` and subtract it. Taking the divergence of
both sides, and using `∇·(divergence-free) = 0`:

```
∇·u** = ∇²φ
```

which is a **Poisson equation**. Solve it for `φ`, then subtract its gradient.
Identifying `φ = Δt·p/ρ` recovers the pressure formulation:

```
∇²p = (ρ/Δt) ∇·u**          solve for p
u^{n+1} = u** − (Δt/ρ) ∇p    subtract
```

Both constants appear verbatim in [`project.go`](project.go):

```go
scale := e.Cfg.Density / (e.Cfg.DT * m.H)   // computeDivergence
scale := e.Cfg.DT / (e.Cfg.Density * m.H)   // subtractGradient
```

This is the step that makes pressure "whatever it must be" — the projection is
literally an orthogonal projection onto the space of divergence-free fields.

---

# Part II — Discretization

## 3. The staggered MAC grid

AeroFlux uses a **Marker-and-Cell** staggered grid (Harlow & Welch, 1965).
Pressure sits at cell centers; each velocity component sits on the cell face
normal to its own axis:

```
                    w(i,j,k+1)
                        ↑
              ┌─────────┴─────────┐
              │                   │
   u(i,j,k) → │    p(i,j,k) ·     │ → u(i+1,j,k)
              │                   │
              └─────────┬─────────┘
                        ↑
                    v(i,j,k)

  p  cell center      u  x-normal faces
  v  y-normal faces   w  z-normal faces
```

Consequently `u` has one *more* plane along x than `p`, and likewise for `v`
along y and `w` along z — visible in [`grid.go`](grid.go):

```go
m.P = fieldLayout(&off, cx,   cy,   cz)
m.U = fieldLayout(&off, cx+1, cy,   cz)
m.V = fieldLayout(&off, cx,   cy+1, cz)
m.W = fieldLayout(&off, cx,   cy,   cz+1)
```

### Why staggering is mandatory, not stylistic

On a **collocated** grid (everything at cell centers), a central-difference
pressure gradient reads `(p[i+1] − p[i−1])/2h` — it skips `p[i]` entirely. Odd
and even cells therefore decouple into two independent subgrids. A pressure
field oscillating `+1, −1, +1, −1` between them has *identically zero* discrete
gradient: it is invisible to the momentum equation, sits in the null space of
the discrete operator, and grows without bound. This is the notorious
**checkerboard instability**.

Staggering eliminates it. Velocities live *between* pressure cells, so the
gradient is the compact two-point difference `(p[i] − p[i−1])/h`, which involves
adjacent cells and cannot be blind to an oscillation between them.

The deeper property: on a staggered grid the discrete divergence and discrete
gradient are **exact adjoints** (negative transposes) of each other. That makes
the discrete Laplacian `∇·∇` symmetric positive-definite — which is what lets
the direct solver use Cholesky factorization (Section 5.2) instead of a general
LU.

### Divergence and gradient stencils

Divergence at a cell center is a difference of the two opposing face velocities
per axis — the faces are already exactly where they need to be:

```
(∇·u)[i,j,k] = ( u[i+1] − u[i] + v[j+1] − v[j] + w[k+1] − w[k] ) / h
```

The gradient reuses the *same* faces. That symmetry is load-bearing: using a
wider or offset stencil here would break adjointness and leave residual
divergence no amount of solver iteration could remove.

### Velocity reconstruction at faces

Only one component lives at any given face, so advection must interpolate the
other two. For a u-face, `v` and `w` are each averaged from the four surrounding
faces of their kind (`faceVelocity` in [`stencil.go`](stencil.go)):

```go
vy = 0.25 * (v[x-1,y,z] + v[x,y,z] + v[x-1,y+1,z] + v[x,y+1,z])
```

This four-point average is the standard MAC reconstruction and keeps the
interpolation second-order accurate.

## 4. Boundary conditions

**Velocity — no-slip solid walls.** Normal components sit exactly *on* the wall,
so they are set to zero directly. Tangential components have no grid point on
the wall, so they are reflected into the ghost layer:

```
u_ghost = −u_interior
```

Averaging across the wall then gives `(u_interior + (−u_interior))/2 = 0`, which
is the no-slip condition at the wall face.

**Pressure — homogeneous Neumann.** Solid walls admit no flux, so the
wall-normal pressure gradient vanishes: `∂p/∂n = 0`, implemented by mirroring
interior pressure into the ghost layer.

A consequence worth stating: with Neumann conditions on *every* boundary,
pressure is determined only **up to an additive constant**. The continuous
problem is singular. This is harmless for the physics (only `∇p` is ever used)
but matters for the linear algebra — see Section 5.2.

---

# Part III — The Solvers

## 5. The pressure Poisson equation

Discretizing `∇²p = f` with the 7-point stencil gives, per interior cell:

```
p[i−1] + p[i+1] + p[j−1] + p[j+1] + p[k−1] + p[k+1] − 6p[i,j,k] = h² f[i,j,k]
```

This is a sparse linear system with one equation per cell — the computational
heart of the solver. AeroFlux offers two ways to solve it.

### 5.1 Red-Black Gauss-Seidel (all platforms)

Rearranging for the center cell gives the iterative update:

```
p[i,j,k] ← ( Σ neighbors − h² f[i,j,k] ) / 6
```

which is exactly [`project.go`](project.go):

```go
sum := p[i-1] + p[i+1] + p[i-sy] + p[i+sy] + p[i-sz] + p[i+sz]
p[i] = (sum - h2*dv[...]) / 6.0
```

Plain Gauss-Seidel is inherently sequential — each update uses values its
neighbors just wrote. **Red-Black checkerboard ordering** breaks that
dependency. Color cells by the parity of `i+j+k`:

```
    R   B   R   B          Every neighbor of a red cell
    B   R   B   R          is black, and vice versa —
    R   B   R   B          the 7-point stencil only ever
    B   R   B   R          reaches across one parity.
```

So *all* red cells can be updated simultaneously from black values with no
read-write conflict, then all black from the freshly-updated red. Two hard
barriers per iteration, full parallelism within each.

The inner loop needs no parity test — the starting index is chosen per row and
the loop strides by 2:

```go
start := g
if (start+y+z)&1 != int(col) {
    start++
}
for x := start; x < g+m.NX; x += 2 { ... }
```

**Convergence behavior.** Gauss-Seidel damps high-frequency error quickly and
low-frequency error slowly, with iteration count scaling roughly as `O(N²)` for
an `N³` grid. This is strongly problem-dependent — measured here on a
discontinuous shear layer:

| Sweeps | `max\|div\|` |
|---:|---|
| 60 | 5.1e-03 |
| 200 | 5.0e-05 |
| 600 | 4.1e-09 |

Smooth initial data converges far faster.

### 5.2 Direct Cholesky via Accelerate (Darwin + cgo)

The key observation: **the Poisson operator is structurally constant.** For a
fixed grid, the matrix never changes — only the right-hand side does. So
factorize once at setup, and pay only triangular solves thereafter.

`EnableSparseSolver` assembles the matrix in compressed-sparse-column form
directly in arena memory and hands raw pointers to Accelerate's `SparseFactor`.
Three details matter:

**Symmetry — store only the lower triangle.** The operator is symmetric, so
`SparseLowerTriangle` halves both memory and factorization work.

**Neumann walls reduce the diagonal.** A wall neighbor simply drops out of the
stencil, so the center coefficient counts only existing neighbors — a corner
cell gets 3, not 6:

```go
diag := 0.0
if ii > 0        { diag++ }
if ii < m.NX-1   { diag++ }
// ... etc
```

**Pinning the constant.** As noted in Section 4, all-Neumann boundaries leave
pressure defined only up to a constant, so the matrix is positive *semi*-definite
and singular — Cholesky requires strict positive-definiteness and would fail.
The standard remedy, used here, is to add a small positive value to one diagonal
entry, anchoring that cell and selecting a unique solution:

```go
if col == 0 { diag += 1.0 }
```

Since only `∇p` is ever used, the anchored offset is physically invisible.

**Sign convention.** The assembled matrix is `−∇²` (positive-definite), so the
direct path negates the RHS, while the iterative path works with `+∇²` directly:

```go
rhs[...] = -dv[...]
```

**The payoff, measured** (`TestDirectSolverMatchesIterative`):

| Solver | `max\|div\|` after projection |
|---|---|
| Red-Black, 400 sweeps | 8.7e-06 |
| Direct Cholesky | **2.1e-12** |

Six orders of magnitude better, and allocation-free per step.

## 6. Strang splitting — the palindrome

Splitting a step into per-axis passes introduces error, because the directional
operators do not commute: `e^{A}e^{B} ≠ e^{A+B}` unless `[A,B] = 0`. The
Baker-Campbell-Hausdorff expansion gives

```
e^{ΔtA} e^{ΔtB} = e^{Δt(A+B) + (Δt²/2)[A,B] + O(Δt³)}
```

so naive sequential splitting is only **first-order** accurate — the commutator
term `[A,B]` pollutes at `O(Δt²)` per step.

**Strang splitting** fixes this by making the sequence *symmetric*: apply each
operator at half-step, forward then in reverse:

```
X → Y → Z → [project] → Z → Y → X → [project]
```

Reversing the order flips the sign of the commutator term, so the two halves
cancel it exactly, leaving `O(Δt³)` local error — **second-order** globally.

The reversal is the entire mechanism. Running `X,Y,Z` twice would cancel nothing.
Beyond formal accuracy, the symmetry also reduces directional bias: the
alternative leaves visible axis-aligned artifacts, since whichever axis is
advected last gets systematically different treatment.

### Why there are two projections

Advection does **not** preserve discrete incompressibility. Measured directly
(`max|div|` traced through one step):

```
initial             3.29e+00
after projection    2.58e-12     ← solve is exact
after fwd sweeps    1.53e-01     ← advection destroys it
after mid project   3.36e-14     ← restored
after rev sweeps    1.52e-01     ← destroyed again
```

A palindrome ending on sweeps would hand the caller a field visibly violating
continuity, even though the mid-step solve was perfect. So `Step` closes with a
second projection, making the *observable* state divergence-free — the invariant
callers actually depend on. `TestStepLeavesFieldDivergenceFree` pins it.

### Double buffering

Sweeps read the previous phase and write scratch fields (`Tmp0/1/2`), which are
then swapped in by exchanging *field descriptors*, not copying data:

```go
m.U, m.Tmp0 = m.Tmp0, m.U
```

Three struct assignments instead of a full-grid memcpy. This is also a
correctness requirement, not just an optimization: a semi-Lagrangian backtrace
reads a neighborhood another worker may still be updating, so writing in place
would make results depend on worker timing.

---

# Part IV — Systems Engineering

## 7. The off-heap arena

`NewArena` reserves anonymous memory as `PROT_NONE`, carves out a 2 MiB-aligned
interior, unmaps the alignment slack, then opens *only* the interior read-write:

```
[ guard page ][ usable region, 2 MiB-aligned ][ guard page ]
      ↑                                              ↑
   PROT_NONE                                     PROT_NONE
```

Ordering matters: everything is born inaccessible and only the interior is
opened, so there is never a window in which the guard pages are readable.

**`Base` is a `uintptr`, not an `unsafe.Pointer`.** This is deliberate and
load-bearing — the garbage collector does not trace `uintptr`, which is exactly
what keeps this memory invisible to it. The cost is fully manual lifetime: after
`Close`, every view handed out is dangling.

**Page size is resolved at runtime.** Hardcoding 4096 would under-size every
guard page on Apple Silicon's 16 KiB pages, letting a stencil overread land in
live memory instead of faulting.

**Over-allocate and trim.** To guarantee a 2 MiB-aligned interior exists, the
reservation adds `HugePageSize + 2·PageSize` of slack, locates the aligned
region, then `munmap`s the excess on both ends rather than squatting on address
space.

**`mlock` is best-effort.** It commonly fails on `RLIMIT_MEMLOCK` without
privilege. An unlocked arena is still correct, just pageable — a tail-latency
cost, not a correctness one. Callers needing hard residency check `Locked()`, and
`Config.Lock` turns failure into an explicit error.

## 8. Fault containment

`debug.SetPanicOnFault` converts a guard-page SIGSEGV into a recoverable Go
panic. Two caveats that are easy to get wrong:

- **It is per-goroutine.** Each worker calls it for itself; the package `init`
  only covers the initializing goroutine.
- **It does not contain anything by itself.** An unrecovered panic still kills
  the process. Containment requires `RecoverFault` deferred in each worker —
  what `runGuarded` does.

`RecoverFault` matches on the `runtime.Error` interface and **re-panics anything
else**, so a real bug (nil map write, bad type assertion) keeps propagating
instead of being silently demoted to an error return.

## 9. Determinism

Bit-for-bit reproducibility rests on three properties, none of which is thread
pinning:

1. **Fixed Z-decomposition.** Shard boundaries are a pure function of
   `(NZ, Workers)`; remainder planes go to the lowest-numbered workers. Each
   worker has its **own channel**, so fan-out is routing, not a race — worker
   *i* always executes shard *i*.
2. **Double-buffered phases.** Each phase is a pure function of the previous
   one, so no result depends on whether a neighbor has been updated yet.
3. **Fixed-order reductions.** Floating-point addition is **not associative**:
   `(a+b)+c ≠ a+(b+c)` in general. Summing partials in completion order would
   make results track scheduling noise. `OrderedAccumulator` gives each worker a
   padded slot and combines them by **index**, never by arrival.

`Config.Normalize` also caps `Workers` at `NZ`, since empty shards would make
the decomposition depend on core count.

Verified across 1/2/3/4/8 workers and `GOMAXPROCS` 1–8
(`TestStepIsDeterministicAcrossWorkerCounts`).

## 10. Memory layout and false sharing

**X is the unit-stride axis.** For fixed `(y,z)`, consecutive x indices are
adjacent `float64`s, so every sweep is a linear walk the hardware prefetcher can
follow.

**128-byte padding.** Every multi-writer control block is padded to a 128-byte
coherence stride — Apple Silicon's adjacent-sector prefetch pulls line pairs, and
128 is a safe superset of x86-64's 64 B. Without it, two workers updating
logically independent counters on the same line would ping-pong exclusive
ownership between cores on every write.

[`layout.go`](layout.go) proves this **at compile time** by cross-subtracting
`unsafe.Sizeof`/`Offsetof` against hardcoded literals:

```go
_ = uint(unsafe.Sizeof(PhaseBarrier{})) - 256
_ = 256 - uint(unsafe.Sizeof(PhaseBarrier{}))
```

A negative result is not representable as an unsigned constant, so the build
fails. The literals are spelled out rather than written as `CacheLineSize*2` on
purpose: the padding arrays are *themselves* sized from `CacheLineSize`, so a
derived form would resize with the constant and never fail — an assertion that
proves nothing. Verified by mutation: changing `CacheLineSize` to 64, or
deleting a padding field, breaks the build with an exact error.

## 11. Concurrency primitives

**Phase barriers.** The dispatch join *is* the barrier. This is a deliberate
choice over a spin barrier: a worker that takes a contained fault still
completes its dispatch, whereas it would never reach a spin barrier and would
hang every peer.

Barriers per step: `6 + 2·(2 + 2·GaussSeidelIters)` — six sweeps, plus each
projection's divergence pass, colour sweeps, and gradient pass. Verified
empirically at 14/26/58 dispatches for 1/4/12 iterations. The direct solver
reduces this to `6 + 2·2`.

[`sync.go`](sync.go) also provides standalone primitives:

- **`PhaseBarrier`** — sense-reversing, so it is reusable without a reset phase.
  The last arriver zeroes the counter *before* flipping the sense, since the flip
  is the release signal and a woken worker must not observe a stale count.
- **`TicketGate`** — position-monotonic turnstile. Admission order is fixed at
  issue time rather than by race, so gated sequences are reproducible. `Do`
  releases the gate even if the body panics, so one faulting worker cannot
  deadlock the rest.
- **`OrderedAccumulator`** — the deterministic reduction described in Section 9.

**Spin backoff.** `internal/cpu` provides `Relax()` in assembly — `YIELD` on
ARM64, `PAUSE` on AMD64. It lives in its own package because **cgo-enabled
packages may not contain Go assembly files**, and the Darwin build links
Accelerate via cgo. The spin escalates to `runtime.Gosched` after 64 iterations,
since burning a core while a peer is descheduled actively delays the peer being
waited on.

## 12. The Accelerate bridge

All addresses cross the cgo boundary as `uintptr`, never as Go pointers — cgo's
pointer-passing rules are satisfied by construction, and nothing is pinned or
copied. This is only sound *because* the memory is off-heap: the GC can neither
move nor collect it.

**vForce shims take the count by value.** vForce's own signatures take the
element count **by pointer** (`vvsin(double *y, const double *x, const int *n)`),
a Fortran-derived convention that is easy to get wrong from cgo and that silently
reads garbage as the length. The C shims take it by value and materialize the
pointer on the C stack, so the Go side cannot make that mistake.
`TestVForceCorrectness` verifies against `math.Sin` to 1e-12, which is what
actually proves the convention is wired correctly.

**Compile-time ABI checks.** The C element sizes the CSC layout assumes are
asserted at build time, so a mismatch is a compile error rather than silently
misread matrix structure:

```go
_ = uint(unsafe.Sizeof(C.long(0))) - 8
_ = uint(unsafe.Sizeof(C.int(0))) - 4
```

---

# Part V — Reference

## Platform support

| Target | Arena | Affinity | Vector math |
|---|---|---|---|
| darwin/arm64 + cgo | mmap, mlock, guard pages | QoS + cluster hint (advisory) | Accelerate vForce, Sparse |
| darwin, no cgo | same | QoS only | pure Go |
| linux/{amd64,arm64} | + `MADV_HUGEPAGE` | `sched_setaffinity` (hard) | pure Go |
| other POSIX | degraded, no guards | `LockOSThread` only | pure Go |

**macOS exposes no per-core pinning.** There is no `sched_setaffinity`
equivalent. `THREAD_AFFINITY_POLICY` is an advisory cache-cluster grouping hint
reachable only through the Mach `thread_policy_set` trap (hence cgo), and thread
QoS only biases away from E-cores. Neither guarantees P-core residency — which is
exactly why determinism here is built on the decomposition rather than on thread
placement. Linux gets a hard kernel guarantee.

## Measured performance

Apple M4, `go test -bench .`. Measured on this machine, not projected.

```
BenchmarkStep/16x16x16          1.6 Mcell/s    0 allocs/op
BenchmarkStep/32x32x32          6.1 Mcell/s    0 allocs/op
BenchmarkStep/48x48x48         10.2 Mcell/s    0 allocs/op

BenchmarkStepByWorkers/1        3.4 Mcell/s    0 allocs/op
BenchmarkStepByWorkers/2        6.5 Mcell/s    0 allocs/op
BenchmarkStepByWorkers/4        7.3 Mcell/s    0 allocs/op
BenchmarkStepByWorkers/8        6.7 Mcell/s    0 allocs/op

BenchmarkVecSin              7454 MB/s         0 allocs/op
```

Scaling is 2.1x from 1→4 workers. The regression at 8 is the E-cores — the same
P/E asymmetry macOS will not let us schedule around.

Accuracy after 20 steps with the direct solver: `max|div| = 2.5e-14`.

## Configuration

| Field | Default | Notes |
|---|---|---|
| `NX,NY,NZ` | required | Interior cell counts |
| `Ghost` | 1 | Halo thickness; ≥1 so the stencil needs no bounds branch |
| `H` | `1/max(NX,NY,NZ)` | Uniform cell size |
| `DT` | diffusion-stable | Fixed; adaptive stepping would break reproducibility |
| `Viscosity` | 0 | `ν`; 0 gives inviscid (Euler) flow |
| `Density` | 1.0 | `ρ` |
| `Workers` | `GOMAXPROCS`, capped at `NZ` | |
| `GaussSeidelIters` | 24 | Ignored when the direct solver is enabled |
| `UseSparseSolver` | false | Set by `EnableSparseSolver()` |
| `Lock` | false | Require `mlock`; errors if unavailable |

## Diagnostics

| Method | Returns |
|---|---|
| `MaxDivergence()` | Largest `\|∇·u\|` — the primary correctness signal |
| `KineticEnergy()` | `½∫\|u\|²dV`; drift measures numerical dissipation |
| `MaxVelocity()` | Largest velocity component |
| `CFL()` | Courant number `\|u\|Δt/h` |

All scan in fixed index order, so they are reproducible.

## Tests

35 tests, all passing under `-race`.

```bash
go test ./...            # correctness
go test -race ./...      # concurrency
go test -bench . ./...   # performance
```

Load-bearing ones:

- `TestStepIsDeterministicAcrossWorkerCounts` — bit-for-bit across 1/2/3/4/8 workers
- `TestStepZeroAllocs`, `TestDirectSolveZeroAllocs` — `AllocsPerRun == 0`
- `TestGuardPageFaultIsContained`, `TestGuardPageBelowBase` — faults contained, both ends
- `TestStepLeavesFieldDivergenceFree` — continuity holds after `Step` returns
- `TestPoissonSolveAgainstKnownSolution` — checked against analytic truth, not self-consistency
- `TestVForceCorrectness` — Accelerate matches `math` to 1e-12
- `TestTicketGateSurvivesPanic` — a fault in a gated section still releases the gate
- `TestPhaseBarrierIsReusable` — 5000 rounds; hangs on a sense-reversal bug

## Known limitations

- **Numerical dissipation.** Semi-Lagrangian advection is stable but diffusive;
  vortices decay faster than physically correct. Higher-order (BFECC, MacCormack)
  advection would reduce this.
- **Iterative convergence is problem-dependent.** Discontinuous data needs many
  more sweeps than smooth data (see the table in Section 5.1). Use the direct
  solver on Darwin.
- **Explicit diffusion is conditionally stable.** `Δt ≤ h²/(6ν)`, enforced by
  `Normalize`. Fine grids with high viscosity want an implicit solve, which is
  not implemented.
- **Fixed timestep only.** CFL-adaptive stepping would make results depend on
  accumulated rounding and break reproducibility.
- **Solid walls only.** No inflow/outflow, periodic, or free-surface conditions.
- **No external forcing.** No buoyancy, gravity, or body-force term; flow evolves
  from initial conditions alone.
- **Single-precision path absent.** All fields are `float64`. A `float32` variant
  would roughly double vector throughput at reduced accuracy.
- `go vet` reports `possible misuse of unsafe.Pointer` in the arena. **This is
  expected.** Vet assumes a `uintptr` may hold a stale address of a *GC-managed*
  object; these refer to `mmap` memory the collector never traces, moves, or
  reclaims. Storing `Base` as an `unsafe.Pointer` to silence it would defeat the
  zero-heap guarantee.

## Source layout

| File | Contents |
|---|---|
| `aeroflux.go` | `Config.Normalize`, engine, worker pool, Strang palindrome |
| `arena.go` | mmap/mlock/guard pages, over-allocate-and-trim, fault recovery |
| `grid.go` | MAC staggered field layout, strides, cell indexing |
| `stencil.go` | Semi-Lagrangian advection, diffusion, no-slip walls |
| `project.go` | Divergence, Red-Black Gauss-Seidel, gradient subtraction |
| `sync.go` | Sense-reversing barrier, ticket turnstile, ordered reduction |
| `layout.go` | Compile-time cache-line assertions |
| `accelerate_darwin.go` | cgo: vForce, vDSP, Sparse Cholesky wrappers |
| `sparse_darwin.go` | 7-point Poisson matrix assembly in CSC form |
| `sys_*.go`, `pin_*.go` | Per-platform syscalls and thread affinity |
| `internal/cpu/` | `YIELD`/`PAUSE` spin hints in assembly |

## Citing AeroFlux

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff) (Citation File
Format 1.2.0). GitHub renders a "Cite this repository" button from it, and
`cffconvert` will emit BibTeX, APA, or Zenodo JSON:

```bibtex
@misc{aeroflux,
  author = {Saini, Eklavya},
  title  = {AeroFlux},
  year   = {2026}
}
```

## References

- Harlow & Welch (1965), *Numerical Calculation of Time-Dependent Viscous
  Incompressible Flow* — the MAC staggered grid.
- Chorin (1968), *Numerical Solution of the Navier-Stokes Equations* — the
  projection method.
- Strang (1968), *On the Construction and Comparison of Difference Schemes* —
  symmetric operator splitting.
- Stam (1999), *Stable Fluids* — semi-Lagrangian advection.
- Bridson (2015), *Fluid Simulation for Computer Graphics*, 2nd ed. — the
  standard practical reference for all of the above.
