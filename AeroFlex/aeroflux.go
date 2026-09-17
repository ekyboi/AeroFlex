package aeroflux

import (
	"errors"
	"fmt"
	"math"
	"runtime"
	"runtime/debug"
	"sync"
)

// Config describes an engine instance. Call Normalize before use; it fills
// defaults and rejects geometries the solver cannot represent.
type Config struct {
	// NX, NY, NZ are interior cell counts.
	NX, NY, NZ int
	// Ghost is halo thickness. Must be >= 1 so the 7-point stencil can
	// read a neighbor plane without a bounds branch in the inner loop.
	Ghost int
	// H is uniform cell size.
	H float64
	// Viscosity is the kinematic viscosity nu.
	Viscosity float64
	// Density is the fluid density rho, used to scale the pressure
	// gradient during projection.
	Density float64
	// DT is the fixed timestep. Fixed rather than adaptive because a
	// CFL-adaptive dt would make results depend on accumulated rounding,
	// destroying bit-for-bit reproducibility across runs.
	DT float64
	// Workers is the worker-goroutine count. Defaults to GOMAXPROCS.
	Workers int
	// GaussSeidelIters is the number of Red-Black sweep pairs per
	// projection when using the iterative solver.
	GaussSeidelIters int
	// UseSparseSolver selects Accelerate's Cholesky direct solve over the
	// iterative Red-Black sweeps. Darwin+cgo only.
	UseSparseSolver bool
	// Lock requests mlock on the arena.
	Lock bool
}

// Normalize fills defaults and validates. It returns a descriptive error
// for any geometry the solver cannot handle rather than clamping silently,
// because a silently-adjusted grid produces physically wrong results that
// look plausible.
func (c *Config) Normalize() error {
	if c.NX <= 0 || c.NY <= 0 || c.NZ <= 0 {
		return fmt.Errorf("aeroflux: interior extents must be positive, got %dx%dx%d", c.NX, c.NY, c.NZ)
	}
	if c.Ghost == 0 {
		c.Ghost = 1
	}
	if c.Ghost < 1 {
		return fmt.Errorf("aeroflux: ghost thickness must be >= 1, got %d", c.Ghost)
	}
	if c.H == 0 {
		c.H = 1.0 / float64(maxInt(c.NX, maxInt(c.NY, c.NZ)))
	}
	if c.H <= 0 || math.IsNaN(c.H) || math.IsInf(c.H, 0) {
		return fmt.Errorf("aeroflux: cell size must be finite and positive, got %v", c.H)
	}
	if c.Density == 0 {
		c.Density = 1.0
	}
	if c.Density <= 0 || math.IsNaN(c.Density) || math.IsInf(c.Density, 0) {
		return fmt.Errorf("aeroflux: density must be finite and positive, got %v", c.Density)
	}
	if c.Viscosity < 0 || math.IsNaN(c.Viscosity) || math.IsInf(c.Viscosity, 0) {
		return fmt.Errorf("aeroflux: viscosity must be finite and non-negative, got %v", c.Viscosity)
	}
	if c.DT == 0 {
		// Default to a diffusion-stable step when viscous, else a mild
		// advective step. Explicit diffusion in 3D is stable for
		// dt <= h^2 / (6*nu); take half that for margin.
		if c.Viscosity > 0 {
			c.DT = 0.5 * c.H * c.H / (6 * c.Viscosity)
		} else {
			c.DT = 0.25 * c.H
		}
	}
	if c.DT <= 0 || math.IsNaN(c.DT) || math.IsInf(c.DT, 0) {
		return fmt.Errorf("aeroflux: timestep must be finite and positive, got %v", c.DT)
	}
	if c.Workers == 0 {
		c.Workers = runtime.GOMAXPROCS(0)
	}
	if c.Workers < 1 {
		return fmt.Errorf("aeroflux: worker count must be >= 1, got %d", c.Workers)
	}
	// More workers than Z-planes would leave workers with empty shards and
	// make the decomposition depend on core count, so cap it.
	if c.Workers > c.NZ {
		c.Workers = c.NZ
	}
	if c.GaussSeidelIters == 0 {
		c.GaussSeidelIters = 24
	}
	if c.GaussSeidelIters < 1 {
		return fmt.Errorf("aeroflux: gauss-seidel iterations must be >= 1, got %d", c.GaussSeidelIters)
	}

	// Explicit diffusion stability. Violating this makes the simulation
	// blow up within a few steps, so refuse rather than produce NaNs.
	if c.Viscosity > 0 {
		limit := c.H * c.H / (6 * c.Viscosity)
		if c.DT > limit {
			return fmt.Errorf("aeroflux: dt %g exceeds explicit diffusion stability limit %g (h=%g, nu=%g); reduce dt or h",
				c.DT, limit, c.H, c.Viscosity)
		}
	}
	return nil
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// shard is one worker's contiguous Z-range. Decomposition is along Z only,
// so each worker's slab is contiguous in memory along the two faster axes
// and the split points are a pure function of (NZ, Workers) -- never of
// scheduling.
type shard struct {
	z0, z1 int // half-open interior z-range
	id     int
	_      [CacheLineSize - 3*8]byte
}

// Engine owns the arena, grid, worker pool and barriers.
type Engine struct {
	Cfg  Config
	Mesh *MAC

	arena   *Arena
	shards  []shard
	wg      sync.WaitGroup
	work    []chan func(shard)
	done    []chan error
	sparse  *PoissonSolver
	started bool

	// matArena holds the CSC matrix arrays and the dense RHS scratch for
	// the direct solver. It is separate from the field arena because
	// Accelerate borrows the matrix pointers for the factorization's whole
	// lifetime, so they must not share space with per-step scratch.
	matArena *Arena
	rhsOff   int

	// curAxis and curDT carry the current phase's parameters to workers
	// through the engine rather than through a capturing closure, which
	// would allocate on every dispatch. They are written by the driver
	// only while no dispatch is in flight, and read by workers only during
	// one, so the dispatch join orders every access.
	curAxis axis
	curDT   float64

	// sweepFn and the solver closures are built once in start() so the hot
	// path sends an already-allocated function value.
	sweepFn func(shard)
	divFn   func(shard)
	gradFn  func(shard)
	redFn   func(shard)
	blackFn func(shard)

	// stepErr collects the first contained fault from any worker in the
	// current step.
	stepErrMu sync.Mutex
	stepErr   error
}

// New builds an engine: sizes and maps the arena, lays out the staggered
// grid, partitions Z across workers, and starts the pool.
func New(cfg Config) (*Engine, error) {
	if err := cfg.Normalize(); err != nil {
		return nil, err
	}

	need := MACBytes(cfg.NX, cfg.NY, cfg.NZ, cfg.Ghost)
	if need <= 0 {
		return nil, errors.New("aeroflux: could not size grid")
	}

	a, err := NewArena(need)
	if err != nil {
		return nil, err
	}
	if cfg.Lock && !a.Locked() {
		// Not fatal, but the caller asked for residency and did not get
		// it; surfacing this beats silently accepting page faults in a
		// latency-sensitive run.
		_ = a.Close()
		return nil, errors.New("aeroflux: mlock requested but unavailable (raise RLIMIT_MEMLOCK)")
	}

	mesh, err := NewMAC(a, cfg.NX, cfg.NY, cfg.NZ, cfg.Ghost, cfg.H)
	if err != nil {
		_ = a.Close()
		return nil, err
	}
	mesh.Zero()

	e := &Engine{
		Cfg:   cfg,
		Mesh:  mesh,
		arena: a,
	}
	e.partition()
	e.start()
	return e, nil
}

// partition splits interior Z across workers. Remainder planes go to the
// lowest-numbered workers so the split is a deterministic function of the
// dimensions alone.
func (e *Engine) partition() {
	n := e.Cfg.Workers
	e.shards = make([]shard, n)
	base := e.Cfg.NZ / n
	rem := e.Cfg.NZ % n
	z := e.Cfg.Ghost
	for i := 0; i < n; i++ {
		cnt := base
		if i < rem {
			cnt++
		}
		e.shards[i] = shard{z0: z, z1: z + cnt, id: i}
		z += cnt
	}
}

// start launches the worker pool. Each worker locks itself to an OS thread,
// applies platform affinity, and enables per-goroutine fault trapping.
//
// Every worker gets its OWN work channel rather than sharing one. A shared
// channel does not route: sending N items for N workers lets a single fast
// worker take two while another takes none, so a shard silently goes
// unprocessed and the dispatcher blocks forever waiting on its done slot.
// Per-worker channels make the fan-out a routing operation instead of a
// race, which is also what makes the Z-decomposition deterministic --
// shard i is always executed by worker i.
func (e *Engine) start() {
	// Built once, reused forever: each is a method value over e with no
	// per-call captures, so dispatching them allocates nothing.
	e.sweepFn = func(s shard) { e.advectDiffuseAxis(s, e.curAxis, e.curDT) }
	e.divFn = e.computeDivergence
	e.gradFn = e.subtractGradient
	e.redFn = func(s shard) { e.gaussSeidelSweep(s, ColorRed) }
	e.blackFn = func(s shard) { e.gaussSeidelSweep(s, ColorBlack) }

	e.work = make([]chan func(shard), e.Cfg.Workers)
	e.done = make([]chan error, e.Cfg.Workers)
	for i := range e.done {
		e.work[i] = make(chan func(shard))
		e.done[i] = make(chan error, 1)
	}
	e.wg.Add(e.Cfg.Workers)
	for i := 0; i < e.Cfg.Workers; i++ {
		go e.worker(e.shards[i])
	}
	e.started = true
}

func (e *Engine) worker(s shard) {
	defer e.wg.Done()

	// Per-goroutine: the package init only covers the initializing
	// goroutine, so each worker must opt in for its own faults to be
	// trappable.
	debug.SetPanicOnFault(true)

	pinWorkerForShard(s.id)
	defer UnpinWorker()

	for fn := range e.work[s.id] {
		e.runGuarded(s, fn)
	}
}

// runGuarded executes one unit of shard work with fault containment. A
// guard-page fault from a blown-up cell is converted to an error and
// recorded; the worker survives and the barrier it would have reached is
// still satisfied, so peers do not deadlock behind it.
func (e *Engine) runGuarded(s shard, fn func(shard)) {
	var ferr error
	defer func() {
		if ferr != nil {
			e.recordErr(ferr)
		}
		e.done[s.id] <- ferr
	}()
	defer RecoverFault(&ferr)
	fn(s)
}

func (e *Engine) recordErr(err error) {
	e.stepErrMu.Lock()
	if e.stepErr == nil {
		e.stepErr = err
	}
	e.stepErrMu.Unlock()
}

// dispatch runs fn on every shard and waits for all of them. The join is
// itself the phase barrier: it returns only once every worker has finished,
// so the next phase cannot read a plane a peer is still writing.
//
// Errors are collected from every worker, not just the first, so one
// worker's contained fault cannot leave another's done slot undrained and
// wedge the following phase.
func (e *Engine) dispatch(fn func(shard)) error {
	for i := 0; i < e.Cfg.Workers; i++ {
		e.work[i] <- fn
	}
	var first error
	for i := 0; i < e.Cfg.Workers; i++ {
		if err := <-e.done[i]; err != nil && first == nil {
			first = err
		}
	}
	return first
}

// Step advances the simulation by one timestep using a symmetric Strang
// splitting palindrome.
//
// The sequence is X, Y, Z, [projection], Z, Y, X, [projection] with
// half-steps on each pass. Applying the axes in one fixed order (X,Y,Z) and
// then again in the reverse order cancels the leading commutator error term
// between the non-commuting per-axis operators, lifting the splitting from
// first to second order in dt. The reversal is the entire point: doing
// X,Y,Z twice would not cancel anything.
//
// There are two projections, not one, and the closing projection is not
// optional. Advection does not preserve discrete incompressibility: each
// semi-Lagrangian sweep reintroduces divergence (measured at ~1.5e-1 for a
// unit-amplitude field here, against ~1e-13 immediately after a solve). A
// palindrome that ended on sweeps would therefore leave the caller looking
// at a field that visibly violates continuity, even though the mid-step
// solve was exact. Projecting last makes the observable state after Step
// divergence-free, which is the invariant callers actually depend on.
//
// Every phase is separated by a hard barrier (the dispatch join), so no
// worker can read a neighbor plane another worker is still writing. See
// SweepBarriersPerStep for how the per-step barrier count breaks down.
func (e *Engine) Step() error {
	e.stepErrMu.Lock()
	e.stepErr = nil
	e.stepErrMu.Unlock()

	e.curDT = e.Cfg.DT * 0.5

	// Each dispatch is itself a hard phase barrier: it returns only once
	// every worker has finished its shard, so the next phase cannot begin
	// while a neighbor plane is still being written. Using the join as the
	// barrier -- rather than an in-worker PhaseBarrier.Wait -- is what
	// keeps a contained fault from deadlocking the step: a worker that
	// faults still completes its dispatch, whereas it would never have
	// reached a spin barrier, hanging every peer.
	//
	// The per-axis parameters travel in engine fields and the closures are
	// built once in start(), rather than being captured per phase. A
	// closure that captures a loop variable must be heap-allocated when it
	// is sent over a channel, which would put six allocations in every
	// single step and break the zero-allocation invariant outright.
	//
	// Boundary conditions are refreshed between phases because each sweep
	// writes interior planes that the next sweep reads through the halo.
	//
	// Forward half-sweeps: X, Y, Z.
	for _, ax := range [...]axis{axisX, axisY, axisZ} {
		e.applyVelocityBC()
		e.curAxis = ax
		if err := e.dispatch(e.sweepFn); err != nil {
			return err
		}
		e.commitVelocity()
	}

	// Pressure projection enforces incompressibility at the palindrome's
	// center, where it sees the symmetric average of both half-sweeps.
	if err := e.project(); err != nil {
		return err
	}

	// Reverse half-sweeps: Z, Y, X. The mirrored order is what cancels the
	// leading splitting-error commutator.
	for _, ax := range [...]axis{axisZ, axisY, axisX} {
		e.applyVelocityBC()
		e.curAxis = ax
		if err := e.dispatch(e.sweepFn); err != nil {
			return err
		}
		e.commitVelocity()
	}

	// Closing projection: restores incompressibility after the reverse
	// sweeps so the field the caller observes satisfies continuity.
	if err := e.project(); err != nil {
		return err
	}

	e.stepErrMu.Lock()
	err := e.stepErr
	e.stepErrMu.Unlock()
	return err
}

// Close stops the pool and releases the arena. After Close every view
// previously obtained from the grid is dangling.
//
// Only the work channel is closed. Workers select on it and return when it
// closes, which is a single unambiguous shutdown signal; closing a separate
// stop channel as well would race -- a worker could take the stop branch
// while a dispatcher was mid-send on work, leaving that send blocked
// forever with no receiver.
func (e *Engine) Close() error {
	if e.started {
		for i := range e.work {
			close(e.work[i])
		}
		e.wg.Wait()
		e.started = false
	}
	if e.sparse != nil {
		e.sparse.Close()
		e.sparse = nil
	}
	if e.matArena != nil {
		// Released only after the factorization above, which borrows these
		// pages: unmapping them first would leave Accelerate holding
		// dangling pointers.
		_ = e.matArena.Close()
		e.matArena = nil
	}
	return e.arena.Close()
}
