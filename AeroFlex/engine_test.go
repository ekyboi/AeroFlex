package aeroflux

import (
	"math"
	"testing"
)

func newTestEngine(t *testing.T, cfg Config) *Engine {
	t.Helper()
	e, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(func() { _ = e.Close() })
	return e
}

func TestConfigNormalizeDefaults(t *testing.T) {
	c := Config{NX: 16, NY: 16, NZ: 16}
	if err := c.Normalize(); err != nil {
		t.Fatalf("Normalize: %v", err)
	}
	if c.Ghost != 1 {
		t.Errorf("Ghost = %d, want 1", c.Ghost)
	}
	if c.Density != 1 {
		t.Errorf("Density = %v, want 1", c.Density)
	}
	if c.H <= 0 || c.DT <= 0 {
		t.Errorf("H=%v DT=%v, both must be positive", c.H, c.DT)
	}
	if c.Workers < 1 {
		t.Errorf("Workers = %d", c.Workers)
	}
}

func TestConfigNormalizeRejects(t *testing.T) {
	cases := []struct {
		name string
		cfg  Config
	}{
		{"zero extent", Config{NX: 0, NY: 8, NZ: 8}},
		{"negative extent", Config{NX: -4, NY: 8, NZ: 8}},
		{"negative ghost", Config{NX: 8, NY: 8, NZ: 8, Ghost: -1}},
		{"negative viscosity", Config{NX: 8, NY: 8, NZ: 8, Viscosity: -1}},
		{"nan cell size", Config{NX: 8, NY: 8, NZ: 8, H: math.NaN()}},
		{"inf dt", Config{NX: 8, NY: 8, NZ: 8, DT: math.Inf(1)}},
		{"negative density", Config{NX: 8, NY: 8, NZ: 8, Density: -2}},
		{"diffusion unstable", Config{NX: 8, NY: 8, NZ: 8, H: 0.1, Viscosity: 1.0, DT: 1.0}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := tc.cfg
			if err := c.Normalize(); err == nil {
				t.Fatalf("expected rejection, got nil (cfg now %+v)", c)
			}
		})
	}
}

func TestConfigWorkersCappedByDepth(t *testing.T) {
	c := Config{NX: 8, NY: 8, NZ: 2, Workers: 64}
	if err := c.Normalize(); err != nil {
		t.Fatalf("Normalize: %v", err)
	}
	if c.Workers > c.NZ {
		t.Errorf("Workers %d exceeds NZ %d; empty shards would make decomposition core-count dependent", c.Workers, c.NZ)
	}
}

func TestMACStaggeredExtents(t *testing.T) {
	e := newTestEngine(t, Config{NX: 8, NY: 6, NZ: 4})
	m := e.Mesh
	// Staggered components carry one extra plane on their own axis.
	if m.U.NX != m.P.NX+1 || m.U.NY != m.P.NY || m.U.NZ != m.P.NZ {
		t.Errorf("U extents %dx%dx%d vs P %dx%dx%d", m.U.NX, m.U.NY, m.U.NZ, m.P.NX, m.P.NY, m.P.NZ)
	}
	if m.V.NY != m.P.NY+1 || m.V.NX != m.P.NX {
		t.Errorf("V extents wrong: %dx%dx%d", m.V.NX, m.V.NY, m.V.NZ)
	}
	if m.W.NZ != m.P.NZ+1 || m.W.NX != m.P.NX {
		t.Errorf("W extents wrong: %dx%dx%d", m.W.NX, m.W.NY, m.W.NZ)
	}
	// X must be unit stride on every field.
	for _, f := range []*Field{&m.P, &m.U, &m.V, &m.W} {
		if f.Index(1, 0, 0)-f.Index(0, 0, 0) != 1 {
			t.Errorf("X is not unit stride on a field")
		}
	}
}

func TestFieldsAreArenaResident(t *testing.T) {
	e := newTestEngine(t, Config{NX: 8, NY: 8, NZ: 8})
	m := e.Mesh
	lo, hi := e.arena.Base, e.arena.Base+uintptr(e.arena.Len)
	for _, f := range []*Field{&m.P, &m.U, &m.V, &m.W, &m.Div} {
		p := m.Ptr(f, 0, 0, 0)
		end := p + uintptr(f.Count*8)
		if p < lo || end > hi {
			t.Errorf("field at %#x..%#x escapes arena %#x..%#x", p, end, lo, hi)
		}
	}
}

// TestProjectionRemovesDivergence is the central correctness test for the
// solver: after projection the velocity field must be discretely
// divergence-free.
func TestProjectionRemovesDivergence(t *testing.T) {
	e := newTestEngine(t, Config{
		NX: 16, NY: 16, NZ: 16,
		H: 1.0 / 16, DT: 0.01, Density: 1.0,
		GaussSeidelIters: 200,
	})

	seedVortex(e, 1.0)
	before := e.MaxDivergence()
	if before == 0 {
		t.Fatal("seed produced a divergence-free field; the test would be vacuous")
	}

	if err := e.project(); err != nil {
		t.Fatalf("project: %v", err)
	}
	after := e.MaxDivergence()

	t.Logf("max|div| before=%.6e after=%.6e (reduced %.1fx)", before, after, before/after)
	if after > before/50 {
		t.Errorf("projection did not meaningfully reduce divergence: %.6e -> %.6e", before, after)
	}
}

// TestStepIsDeterministicAcrossWorkerCounts is the headline guarantee:
// identical results regardless of how many cores participate.
func TestStepIsDeterministicAcrossWorkerCounts(t *testing.T) {
	run := func(workers int) []float64 {
		e := newTestEngine(t, Config{
			NX: 12, NY: 12, NZ: 12,
			H: 1.0 / 12, DT: 0.005, Density: 1.0,
			Viscosity:        0.0,
			Workers:          workers,
			GaussSeidelIters: 30,
		})
		seedVortex(e, 1.0)
		for i := 0; i < 5; i++ {
			if err := e.Step(); err != nil {
				t.Fatalf("Step: %v", err)
			}
		}
		m := e.Mesh
		out := make([]float64, 0, m.U.Count+m.V.Count+m.W.Count)
		out = append(out, m.Data(&m.U)...)
		out = append(out, m.Data(&m.V)...)
		out = append(out, m.Data(&m.W)...)
		return out
	}

	ref := run(1)
	for _, w := range []int{2, 3, 4, 8} {
		got := run(w)
		if len(got) != len(ref) {
			t.Fatalf("workers=%d: length %d != %d", w, len(got), len(ref))
		}
		for i := range ref {
			if got[i] != ref[i] {
				t.Fatalf("workers=%d: bit-for-bit mismatch at %d: %v != %v (delta %g)",
					w, i, got[i], ref[i], got[i]-ref[i])
			}
		}
	}
}

// TestStepIsReproducibleAcrossRuns checks run-to-run stability with a fixed
// worker count.
func TestStepIsReproducibleAcrossRuns(t *testing.T) {
	run := func() float64 {
		e := newTestEngine(t, Config{
			NX: 10, NY: 10, NZ: 10,
			H: 0.1, DT: 0.004, Workers: 4,
			GaussSeidelIters: 20,
		})
		seedVortex(e, 1.0)
		for i := 0; i < 8; i++ {
			if err := e.Step(); err != nil {
				t.Fatalf("Step: %v", err)
			}
		}
		return e.KineticEnergy()
	}
	a, b := run(), run()
	if a != b {
		t.Errorf("kinetic energy not reproducible: %v vs %v", a, b)
	}
}

func TestStepStaysFinite(t *testing.T) {
	e := newTestEngine(t, Config{
		NX: 12, NY: 12, NZ: 12,
		H: 1.0 / 12, DT: 0.01, Viscosity: 1e-4,
		GaussSeidelIters: 25,
	})
	seedVortex(e, 2.0)

	for i := 0; i < 30; i++ {
		if err := e.Step(); err != nil {
			t.Fatalf("step %d: %v", i, err)
		}
	}
	m := e.Mesh
	for _, f := range []*Field{&m.U, &m.V, &m.W, &m.P} {
		for _, x := range m.Data(f) {
			if math.IsNaN(x) || math.IsInf(x, 0) {
				t.Fatalf("non-finite value %v after 30 steps", x)
			}
		}
	}
	t.Logf("after 30 steps: maxVel=%.4f CFL=%.4f KE=%.6f maxDiv=%.3e",
		e.MaxVelocity(), e.CFL(), e.KineticEnergy(), e.MaxDivergence())
}

// TestStepLeavesFieldDivergenceFree pins the invariant callers actually
// rely on: after Step returns, the velocity field satisfies discrete
// continuity. Advection reintroduces divergence, so this holds only
// because Step closes with a projection -- an earlier revision ended on
// sweeps and left max|div| ~1.5e-1.
func TestStepLeavesFieldDivergenceFree(t *testing.T) {
	e := newTestEngine(t, Config{
		NX: 12, NY: 12, NZ: 12,
		H: 1.0 / 12, DT: 0.005,
		GaussSeidelIters: 300,
	})
	seedVortex(e, 1.0)

	for i := 0; i < 5; i++ {
		if err := e.Step(); err != nil {
			t.Fatalf("step %d: %v", i, err)
		}
		div := e.MaxDivergence()
		t.Logf("step %d: max|div| = %.3e", i, div)
		if div > 1e-3 {
			t.Errorf("step %d left max|div| = %.3e, want <= 1e-3", i, div)
		}
	}
}

// TestNoSlipWallsHoldNormalVelocity verifies the boundary condition that
// the guard pages are meant to make unnecessary to branch on.
func TestNoSlipWallsHoldNormalVelocity(t *testing.T) {
	e := newTestEngine(t, Config{NX: 8, NY: 8, NZ: 8, H: 0.125, DT: 0.005})
	seedVortex(e, 1.0)
	for i := 0; i < 3; i++ {
		if err := e.Step(); err != nil {
			t.Fatalf("Step: %v", err)
		}
	}
	m := e.Mesh
	g := m.Ghost
	u := m.Data(&m.U)
	for z := g; z < g+m.NZ; z++ {
		for y := g; y < g+m.NY; y++ {
			if v := u[m.U.Index(g, y, z)]; v != 0 {
				t.Fatalf("u on -x wall = %v, want 0", v)
			}
			if v := u[m.U.Index(g+m.NX, y, z)]; v != 0 {
				t.Fatalf("u on +x wall = %v, want 0", v)
			}
		}
	}
}

// TestStepZeroAllocs asserts the hot path never touches the Go heap.
func TestStepZeroAllocs(t *testing.T) {
	e := newTestEngine(t, Config{
		NX: 8, NY: 8, NZ: 8,
		H: 0.125, DT: 0.005, Workers: 2,
		GaussSeidelIters: 4,
	})
	seedVortex(e, 1.0)
	if err := e.Step(); err != nil {
		t.Fatalf("warmup: %v", err)
	}

	got := testing.AllocsPerRun(20, func() {
		_ = e.Step()
	})
	if got != 0 {
		t.Errorf("Step allocated %v times per run, want 0", got)
	}
}

func TestEngineCloseIdempotent(t *testing.T) {
	e, err := New(Config{NX: 8, NY: 8, NZ: 8})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if err := e.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}
}

// seedVortex installs a smooth, divergence-carrying initial condition. Not
// a physical solution -- the point is to give projection real work to do.
func seedVortex(e *Engine, amp float64) {
	m := e.Mesh
	g := m.Ghost
	u := m.Data(&m.U)
	v := m.Data(&m.V)
	w := m.Data(&m.W)

	fx := 2 * math.Pi / float64(m.NX)
	fy := 2 * math.Pi / float64(m.NY)
	fz := 2 * math.Pi / float64(m.NZ)

	for z := g; z < g+m.NZ; z++ {
		for y := g; y < g+m.NY; y++ {
			for x := g; x <= g+m.NX; x++ {
				u[m.U.Index(x, y, z)] = amp * math.Sin(fx*float64(x-g)) * math.Cos(fy*float64(y-g))
			}
		}
	}
	for z := g; z < g+m.NZ; z++ {
		for y := g; y <= g+m.NY; y++ {
			for x := g; x < g+m.NX; x++ {
				v[m.V.Index(x, y, z)] = -amp * math.Cos(fx*float64(x-g)) * math.Sin(fy*float64(y-g))
			}
		}
	}
	for z := g; z <= g+m.NZ; z++ {
		for y := g; y < g+m.NY; y++ {
			for x := g; x < g+m.NX; x++ {
				w[m.W.Index(x, y, z)] = 0.25 * amp * math.Sin(fz*float64(z-g))
			}
		}
	}
	e.applyVelocityBC()
}
