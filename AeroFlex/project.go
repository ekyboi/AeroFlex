package aeroflux

import "math"

// project enforces incompressibility.
//
// Helmholtz-Hodge: any vector field splits uniquely into a divergence-free
// part and a gradient. Solving grad^2 p = rho/dt * div(u) and subtracting
// dt/rho * grad p leaves exactly the divergence-free part. This is where
// pressure enters the algorithm -- it is not advanced in time, it is solved
// for afresh each step as the Lagrange multiplier enforcing div(u) = 0.
func (e *Engine) project() error {
	e.applyVelocityBC()

	if err := e.dispatch(e.divFn); err != nil {
		return err
	}

	if e.Cfg.UseSparseSolver && e.sparse != nil {
		if err := e.solvePressureDirect(); err != nil {
			return err
		}
	} else {
		if err := e.solvePressureRedBlack(); err != nil {
			return err
		}
	}

	if err := e.dispatch(e.gradFn); err != nil {
		return err
	}
	e.applyVelocityBC()
	return nil
}

// computeDivergence fills Div with rho/dt * div(u) over this shard.
//
// On a staggered grid the divergence at a cell center is the difference of
// the two opposing face velocities, a compact two-point difference per
// axis. That compactness is precisely what the collocated layout lacks and
// why the checkerboard pressure mode cannot appear here.
func (e *Engine) computeDivergence(s shard) {
	m := e.Mesh
	u := m.Data(&m.U)
	v := m.Data(&m.V)
	w := m.Data(&m.W)
	dv := m.Data(&m.Div)

	scale := e.Cfg.Density / (e.Cfg.DT * m.H)
	g := m.Ghost

	for z := s.z0; z < s.z1; z++ {
		for y := g; y < g+m.NY; y++ {
			for x := g; x < g+m.NX; x++ {
				du := u[m.U.Index(x+1, y, z)] - u[m.U.Index(x, y, z)]
				dvv := v[m.V.Index(x, y+1, z)] - v[m.V.Index(x, y, z)]
				dw := w[m.W.Index(x, y, z+1)] - w[m.W.Index(x, y, z)]
				dv[m.Div.Index(x, y, z)] = scale * (du + dvv + dw)
			}
		}
	}
}

// solvePressureRedBlack runs sharded Red-Black Gauss-Seidel.
//
// The checkerboard split is what makes Gauss-Seidel parallelizable at all:
// the 7-point Laplacian couples a cell only to its six face neighbors, and
// on a checkerboard every neighbor of a red cell is black. So all red cells
// can be updated simultaneously from black values with no read-write
// conflict, then all black from the freshly updated red. Each colour pass
// is a hard barrier; updating both colours at once would reintroduce the
// very data dependence the split removes.
func (e *Engine) solvePressureRedBlack() error {
	m := e.Mesh
	p := m.Data(&m.P)

	// Pressure is solved fresh each step, so start from zero. Warm-starting
	// from the previous step would converge faster but make the result
	// depend on history in a way that complicates exact reproducibility
	// across differing step counts.
	for i := range p {
		p[i] = 0
	}

	// Prebuilt colour closures (see Engine.start) keep the inner loop
	// allocation-free: a captured colour variable would heap-allocate on
	// every one of the 2*GaussSeidelIters dispatches per step.
	for it := 0; it < e.Cfg.GaussSeidelIters; it++ {
		e.applyPressureBC()
		if err := e.dispatch(e.redFn); err != nil {
			return err
		}
		e.applyPressureBC()
		if err := e.dispatch(e.blackFn); err != nil {
			return err
		}
	}
	e.applyPressureBC()
	return nil
}

// gaussSeidelSweep relaxes one colour over this worker's shard.
func (e *Engine) gaussSeidelSweep(s shard, col Color) {
	m := e.Mesh
	p := m.Data(&m.P)
	dv := m.Data(&m.Div)
	h2 := m.H * m.H
	g := m.Ghost

	sy, sz := m.P.SY, m.P.SZ

	for z := s.z0; z < s.z1; z++ {
		for y := g; y < g+m.NY; y++ {
			// Parity of the first interior cell in this row decides where
			// this colour starts; stepping by 2 then visits only cells of
			// the matching colour, so no per-cell parity test is needed in
			// the inner loop.
			start := g
			if (start+y+z)&1 != int(col) {
				start++
			}
			base := m.P.Index(0, y, z)
			for x := start; x < g+m.NX; x += 2 {
				i := base + x
				sum := p[i-1] + p[i+1] + p[i-sy] + p[i+sy] + p[i-sz] + p[i+sz]
				p[i] = (sum - h2*dv[m.Div.Index(x, y, z)]) / 6.0
			}
		}
	}
}

// applyPressureBC imposes homogeneous Neumann (dp/dn = 0) on solid walls by
// mirroring interior pressure into the ghost layer.
//
// Neumann rather than Dirichlet because the walls are solid: no flow
// crosses them, so the pressure gradient normal to them must vanish. A
// consequence is that pressure is determined only up to an additive
// constant, which is harmless since only its gradient is ever used.
func (e *Engine) applyPressureBC() {
	m := e.Mesh
	p := m.Data(&m.P)
	g := m.Ghost
	x1, y1, z1 := g+m.NX, g+m.NY, g+m.NZ

	for z := g; z < z1; z++ {
		for y := g; y < y1; y++ {
			p[m.P.Index(g-1, y, z)] = p[m.P.Index(g, y, z)]
			p[m.P.Index(x1, y, z)] = p[m.P.Index(x1-1, y, z)]
		}
	}
	for z := g; z < z1; z++ {
		for x := g - 1; x <= x1; x++ {
			p[m.P.Index(x, g-1, z)] = p[m.P.Index(x, g, z)]
			p[m.P.Index(x, y1, z)] = p[m.P.Index(x, y1-1, z)]
		}
	}
	for y := g - 1; y <= y1; y++ {
		for x := g - 1; x <= x1; x++ {
			p[m.P.Index(x, y, g-1)] = p[m.P.Index(x, y, g)]
			p[m.P.Index(x, y, z1)] = p[m.P.Index(x, y, z1-1)]
		}
	}
}

// subtractGradient removes dt/rho * grad p from the velocity field, leaving
// it discretely divergence-free.
//
// The gradient is evaluated with the same two-point difference the
// divergence used, on the same faces. Using a wider or offset stencil here
// would break the adjointness of grad and div and leave residual
// divergence that no number of solver iterations could remove.
func (e *Engine) subtractGradient(s shard) {
	m := e.Mesh
	p := m.Data(&m.P)
	u := m.Data(&m.U)
	v := m.Data(&m.V)
	w := m.Data(&m.W)

	scale := e.Cfg.DT / (e.Cfg.Density * m.H)
	g := m.Ghost

	for z := s.z0; z < s.z1; z++ {
		for y := g; y < g+m.NY; y++ {
			for x := g; x <= g+m.NX; x++ {
				u[m.U.Index(x, y, z)] -= scale * (p[m.P.Index(x, y, z)] - p[m.P.Index(x-1, y, z)])
			}
		}
		for y := g; y <= g+m.NY; y++ {
			for x := g; x < g+m.NX; x++ {
				v[m.V.Index(x, y, z)] -= scale * (p[m.P.Index(x, y, z)] - p[m.P.Index(x, y-1, z)])
			}
		}
	}
	zhi := s.z1
	if s.z1 == g+m.NZ {
		zhi++
	}
	for z := s.z0; z < zhi; z++ {
		for y := g; y < g+m.NY; y++ {
			for x := g; x < g+m.NX; x++ {
				w[m.W.Index(x, y, z)] -= scale * (p[m.P.Index(x, y, z)] - p[m.P.Index(x, y, z-1)])
			}
		}
	}
}

// MaxDivergence returns the largest absolute divergence over interior
// cells. After projection this should be near machine epsilon relative to
// the velocity scale; it is the primary correctness signal for the solver.
func (e *Engine) MaxDivergence() float64 {
	m := e.Mesh
	u := m.Data(&m.U)
	v := m.Data(&m.V)
	w := m.Data(&m.W)
	g := m.Ghost
	invH := 1.0 / m.H

	var mx float64
	for z := g; z < g+m.NZ; z++ {
		for y := g; y < g+m.NY; y++ {
			for x := g; x < g+m.NX; x++ {
				d := invH * ((u[m.U.Index(x+1, y, z)] - u[m.U.Index(x, y, z)]) +
					(v[m.V.Index(x, y+1, z)] - v[m.V.Index(x, y, z)]) +
					(w[m.W.Index(x, y, z+1)] - w[m.W.Index(x, y, z)]))
				if a := math.Abs(d); a > mx {
					mx = a
				}
			}
		}
	}
	return mx
}

// KineticEnergy returns the total kinetic energy, summed in a fixed index
// order for reproducibility. With zero viscosity this should be very nearly
// conserved; its drift is the standard measure of a scheme's numerical
// dissipation.
func (e *Engine) KineticEnergy() float64 {
	m := e.Mesh
	u := m.Data(&m.U)
	v := m.Data(&m.V)
	w := m.Data(&m.W)
	g := m.Ghost
	vol := m.H * m.H * m.H

	var sum float64
	for z := g; z < g+m.NZ; z++ {
		for y := g; y < g+m.NY; y++ {
			for x := g; x < g+m.NX; x++ {
				uc := 0.5 * (u[m.U.Index(x, y, z)] + u[m.U.Index(x+1, y, z)])
				vc := 0.5 * (v[m.V.Index(x, y, z)] + v[m.V.Index(x, y+1, z)])
				wc := 0.5 * (w[m.W.Index(x, y, z)] + w[m.W.Index(x, y, z+1)])
				sum += uc*uc + vc*vc + wc*wc
			}
		}
	}
	return 0.5 * sum * vol
}
