//go:build darwin && cgo

package aeroflux

import (
	"math"
	"testing"
)

// TestDirectSolverMatchesIterative checks that Accelerate's Cholesky solve
// and the Red-Black iterative sweeps converge to the same projected field.
//
// They will not agree bit-for-bit: one is a direct factorization and the
// other an iterative approximation, so the comparison is on whether both
// remove divergence to a comparable tolerance.
func TestDirectSolverMatchesIterative(t *testing.T) {
	const (
		nx, ny, nz = 12, 12, 12
		h          = 1.0 / 12
		dt         = 0.01
	)

	iter := newTestEngine(t, Config{
		NX: nx, NY: ny, NZ: nz, H: h, DT: dt,
		GaussSeidelIters: 400,
	})
	seedVortex(iter, 1.0)
	if err := iter.project(); err != nil {
		t.Fatalf("iterative project: %v", err)
	}
	iterDiv := iter.MaxDivergence()

	direct := newTestEngine(t, Config{
		NX: nx, NY: ny, NZ: nz, H: h, DT: dt,
	})
	if err := direct.EnableSparseSolver(); err != nil {
		t.Fatalf("EnableSparseSolver: %v", err)
	}
	seedVortex(direct, 1.0)
	if err := direct.project(); err != nil {
		t.Fatalf("direct project: %v", err)
	}
	directDiv := direct.MaxDivergence()

	t.Logf("max|div| iterative=%.6e direct=%.6e", iterDiv, directDiv)

	if math.IsNaN(directDiv) || math.IsInf(directDiv, 0) {
		t.Fatalf("direct solver produced non-finite divergence %v", directDiv)
	}
	// The direct solve is exact up to roundoff, so it should be at least as
	// good as a long iterative run.
	if directDiv > iterDiv*2 && directDiv > 1e-8 {
		t.Errorf("direct solver divergence %.6e much worse than iterative %.6e", directDiv, iterDiv)
	}
}

// TestDirectSolverStepsStably runs the full palindrome with the direct
// solver engaged.
func TestDirectSolverStepsStably(t *testing.T) {
	e := newTestEngine(t, Config{
		NX: 10, NY: 10, NZ: 10, H: 0.1, DT: 0.005,
	})
	if err := e.EnableSparseSolver(); err != nil {
		t.Fatalf("EnableSparseSolver: %v", err)
	}
	seedVortex(e, 1.0)

	for i := 0; i < 20; i++ {
		if err := e.Step(); err != nil {
			t.Fatalf("step %d: %v", i, err)
		}
	}
	m := e.Mesh
	for _, f := range []*Field{&m.U, &m.V, &m.W, &m.P} {
		for _, x := range m.Data(f) {
			if math.IsNaN(x) || math.IsInf(x, 0) {
				t.Fatalf("non-finite value %v after 20 direct-solver steps", x)
			}
		}
	}
	t.Logf("direct solver after 20 steps: maxVel=%.4f KE=%.6f maxDiv=%.3e",
		e.MaxVelocity(), e.KineticEnergy(), e.MaxDivergence())
}

// TestDirectSolveZeroAllocs confirms the per-step direct solve stays off
// the Go heap.
func TestDirectSolveZeroAllocs(t *testing.T) {
	e := newTestEngine(t, Config{
		NX: 8, NY: 8, NZ: 8, H: 0.125, DT: 0.005, Workers: 2,
	})
	if err := e.EnableSparseSolver(); err != nil {
		t.Fatalf("EnableSparseSolver: %v", err)
	}
	seedVortex(e, 1.0)
	if err := e.Step(); err != nil {
		t.Fatalf("warmup: %v", err)
	}

	got := testing.AllocsPerRun(20, func() {
		_ = e.Step()
	})
	if got != 0 {
		t.Errorf("direct-solver Step allocated %v times per run, want 0", got)
	}
}
