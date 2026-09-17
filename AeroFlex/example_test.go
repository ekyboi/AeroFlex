package aeroflux_test

import (
	"fmt"
	"math"

	"aeroflux"
)

// Example runs a short simulation and reports that the resulting field is
// divergence-free, which is the invariant the projection step exists to
// establish.
func Example() {
	e, err := aeroflux.New(aeroflux.Config{
		NX: 16, NY: 16, NZ: 16,
		H:         1.0 / 16,
		DT:        0.004,
		Viscosity: 1e-5,
		Density:   1.0,
		// A shear layer is a discontinuous initial condition, whose
		// high-frequency content Gauss-Seidel damps slowly: 60 sweeps
		// leave max|div| ~5e-3, while 400 reach ~1e-7. Smooth initial
		// data converges far faster. On Darwin, EnableSparseSolver
		// sidesteps the trade-off entirely with a direct factorization.
		GaussSeidelIters: 400,
	})
	if err != nil {
		fmt.Println("init:", err)
		return
	}
	defer e.Close()

	// Seed a shear layer directly into arena memory.
	m := e.Mesh
	u := m.Data(&m.U)
	g := m.Ghost
	for z := g; z < g+m.NZ; z++ {
		for y := g; y < g+m.NY; y++ {
			val := 1.0
			if y < g+m.NY/2 {
				val = -1.0
			}
			for x := g; x <= g+m.NX; x++ {
				u[m.U.Index(x, y, z)] = val
			}
		}
	}

	for i := 0; i < 10; i++ {
		if err := e.Step(); err != nil {
			fmt.Println("step:", err)
			return
		}
	}

	fmt.Println("finite:", !math.IsNaN(e.KineticEnergy()))
	fmt.Println("divergence-free:", e.MaxDivergence() < 1e-3)
	// Output:
	// finite: true
	// divergence-free: true
}
