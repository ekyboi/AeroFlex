package aeroflux

import "math"

type axis uint8

const (
	axisX axis = iota
	axisY
	axisZ
)

// advectDiffuseAxis applies one directional half-step of advection plus
// diffusion to all three velocity components over this worker's Z-shard.
//
// Results are written to the Tmp fields rather than in place: a
// semi-Lagrangian backtrace reads a neighborhood that another worker may
// still be updating, so writing in place would make the result depend on
// worker timing. Double-buffering makes each phase a pure function of the
// previous phase's state, which is the core of the determinism guarantee.
func (e *Engine) advectDiffuseAxis(s shard, ax axis, dt float64) {
	e.advectComponent(s, &e.Mesh.U, &e.Mesh.Tmp0, compU, ax, dt)
	e.advectComponent(s, &e.Mesh.V, &e.Mesh.Tmp1, compV, ax, dt)
	e.advectComponent(s, &e.Mesh.W, &e.Mesh.Tmp2, compW, ax, dt)
}

type component uint8

const (
	compU component = iota
	compV
	compW
)

// advectComponent transports one velocity component along a single axis and
// applies the matching directional diffusion term.
//
// Advection uses a semi-Lagrangian backtrace: for each face, trace the
// characteristic backward by dt and interpolate the departure value. This
// is unconditionally stable regardless of CFL number, unlike an explicit
// upwind scheme which would cap dt and, on blow-up, would produce NaNs
// rather than the out-of-range index that the guard pages are there to
// catch.
func (e *Engine) advectComponent(s shard, src, dst *Field, c component, ax axis, dt float64) {
	m := e.Mesh
	h := m.H
	invH := 1.0 / h
	nu := e.Cfg.Viscosity

	sd := m.Data(src)
	dd := m.Data(dst)

	// Face-count extents for this component.
	x0, x1 := m.Ghost, m.Ghost+m.NX
	y0, y1 := m.Ghost, m.Ghost+m.NY
	z0, z1 := s.z0, s.z1
	switch c {
	case compU:
		x1++
	case compV:
		y1++
	case compW:
		// The shard owns interior z-planes; the extra w-plane belongs to
		// the last shard so no plane is written twice.
		if s.z1 == m.Ghost+m.NZ {
			z1++
		}
	}

	// Diffusion coefficient for one directional pass. The full Laplacian
	// is split across the three axis passes, so each pass carries only its
	// own second difference.
	lam := nu * dt * invH * invH

	for z := z0; z < z1; z++ {
		for y := y0; y < y1; y++ {
			base := src.Index(0, y, z)
			dbase := dst.Index(0, y, z)
			for x := x0; x < x1; x++ {
				i := base + x

				// Interpolate the full velocity vector at this face.
				vx, vy, vz := e.faceVelocity(c, x, y, z)

				// Backtrace only along the active axis; the palindrome
				// composes the axes.
				var px, py, pz float64
				px, py, pz = float64(x), float64(y), float64(z)
				switch ax {
				case axisX:
					px -= dt * vx * invH
				case axisY:
					py -= dt * vy * invH
				case axisZ:
					pz -= dt * vz * invH
				}

				adv := e.sampleTrilinear(src, sd, px, py, pz)

				// Directional second difference for diffusion.
				var lap float64
				if lam != 0 {
					var im, ip int
					switch ax {
					case axisX:
						im, ip = i-1, i+1
					case axisY:
						im, ip = i-src.SY, i+src.SY
					case axisZ:
						im, ip = i-src.SZ, i+src.SZ
					}
					lap = lam * (sd[im] - 2*sd[i] + sd[ip])
				}

				dd[dbase+x] = adv + lap
			}
		}
	}
}

// faceVelocity reconstructs the full velocity vector at a staggered face.
//
// On a MAC grid only one component lives at any given face, so the other
// two must be averaged in from their own faces. For a u-face, v and w are
// each the mean of the four surrounding faces of their kind; that
// four-point average is the standard MAC reconstruction and keeps the
// interpolation second-order.
func (e *Engine) faceVelocity(c component, x, y, z int) (vx, vy, vz float64) {
	m := e.Mesh
	u := m.Data(&m.U)
	v := m.Data(&m.V)
	w := m.Data(&m.W)

	switch c {
	case compU:
		vx = u[m.U.Index(x, y, z)]
		vy = 0.25 * (v[m.V.Index(x-1, y, z)] + v[m.V.Index(x, y, z)] +
			v[m.V.Index(x-1, y+1, z)] + v[m.V.Index(x, y+1, z)])
		vz = 0.25 * (w[m.W.Index(x-1, y, z)] + w[m.W.Index(x, y, z)] +
			w[m.W.Index(x-1, y, z+1)] + w[m.W.Index(x, y, z+1)])
	case compV:
		vx = 0.25 * (u[m.U.Index(x, y-1, z)] + u[m.U.Index(x+1, y-1, z)] +
			u[m.U.Index(x, y, z)] + u[m.U.Index(x+1, y, z)])
		vy = v[m.V.Index(x, y, z)]
		vz = 0.25 * (w[m.W.Index(x, y-1, z)] + w[m.W.Index(x, y, z)] +
			w[m.W.Index(x, y-1, z+1)] + w[m.W.Index(x, y, z+1)])
	case compW:
		vx = 0.25 * (u[m.U.Index(x, y, z-1)] + u[m.U.Index(x+1, y, z-1)] +
			u[m.U.Index(x, y, z)] + u[m.U.Index(x+1, y, z)])
		vy = 0.25 * (v[m.V.Index(x, y, z-1)] + v[m.V.Index(x, y+1, z-1)] +
			v[m.V.Index(x, y, z)] + v[m.V.Index(x, y+1, z)])
		vz = w[m.W.Index(x, y, z)]
	}
	return
}

// sampleTrilinear interpolates f at fractional padded coordinates.
//
// The clamp keeps the backtrace inside the halo. This is a deliberate
// exception to the "no branches in the hot loop" rule, and it is worth
// stating why: guard pages catch a *catastrophic* excursion, but a
// semi-Lagrangian trace that lands one cell outside the halo during normal
// operation is not catastrophic, it is ordinary near-boundary behavior.
// Faulting on it would turn a routine event into a killed step. The guard
// pages remain the backstop for genuine blow-ups, where the coordinate is
// NaN or wildly out of range and the clamp cannot mask it -- NaN fails
// every comparison, so it propagates to the index and faults, which is
// exactly the behavior we want.
func (e *Engine) sampleTrilinear(f *Field, d []float64, px, py, pz float64) float64 {
	maxX := float64(f.NX - 1)
	maxY := float64(f.NY - 1)
	maxZ := float64(f.NZ - 1)

	if px < 0 {
		px = 0
	} else if px > maxX {
		px = maxX
	}
	if py < 0 {
		py = 0
	} else if py > maxY {
		py = maxY
	}
	if pz < 0 {
		pz = 0
	} else if pz > maxZ {
		pz = maxZ
	}

	x0 := int(px)
	y0 := int(py)
	z0 := int(pz)
	if x0 >= f.NX-1 {
		x0 = f.NX - 2
	}
	if y0 >= f.NY-1 {
		y0 = f.NY - 2
	}
	if z0 >= f.NZ-1 {
		z0 = f.NZ - 2
	}

	tx := px - float64(x0)
	ty := py - float64(y0)
	tz := pz - float64(z0)

	i000 := f.Index(x0, y0, z0)
	i100 := i000 + 1
	i010 := i000 + f.SY
	i110 := i010 + 1
	i001 := i000 + f.SZ
	i101 := i001 + 1
	i011 := i001 + f.SY
	i111 := i011 + 1

	c00 := d[i000]*(1-tx) + d[i100]*tx
	c10 := d[i010]*(1-tx) + d[i110]*tx
	c01 := d[i001]*(1-tx) + d[i101]*tx
	c11 := d[i011]*(1-tx) + d[i111]*tx

	c0 := c00*(1-ty) + c10*ty
	c1 := c01*(1-ty) + c11*ty

	return c0*(1-tz) + c1*tz
}

// commitVelocity swaps the freshly advected Tmp fields back into U, V, W.
// Field descriptors are exchanged rather than data copied, so a phase
// transition costs three struct assignments instead of a full-grid memcpy.
func (e *Engine) commitVelocity() {
	m := e.Mesh
	m.U, m.Tmp0 = m.Tmp0, m.U
	m.V, m.Tmp1 = m.Tmp1, m.V
	m.W, m.Tmp2 = m.Tmp2, m.W
}

// applyVelocityBC enforces no-slip solid walls on the domain boundary.
//
// Normal components sit exactly on the wall and are set to zero. Tangential
// components are reflected so their average across the wall vanishes, which
// is the no-slip condition on a staggered grid.
//
// This runs on the driver goroutine between phases rather than inside a
// worker, because the halo is shared across shard boundaries and having
// each worker write its own halo would race.
func (e *Engine) applyVelocityBC() {
	m := e.Mesh
	g := m.Ghost
	u := m.Data(&m.U)
	v := m.Data(&m.V)
	w := m.Data(&m.W)

	ux0, ux1 := g, g+m.NX // u has NX+1 interior faces: [g, g+NX]
	y0, y1 := g, g+m.NY
	z0, z1 := g, g+m.NZ

	// Normal velocity vanishes on the walls it is normal to.
	for z := z0; z < z1; z++ {
		for y := y0; y < y1; y++ {
			u[m.U.Index(ux0, y, z)] = 0
			u[m.U.Index(ux1, y, z)] = 0
		}
	}
	for z := z0; z < z1; z++ {
		for x := g; x < g+m.NX; x++ {
			v[m.V.Index(x, y0, z)] = 0
			v[m.V.Index(x, y1, z)] = 0
		}
	}
	for y := y0; y < y1; y++ {
		for x := g; x < g+m.NX; x++ {
			w[m.W.Index(x, y, z0)] = 0
			w[m.W.Index(x, y, z1)] = 0
		}
	}

	// Tangential reflection into the ghost layer: u_ghost = -u_interior
	// makes the wall-face average zero, giving no-slip.
	for z := z0; z < z1; z++ {
		for x := ux0; x <= ux1; x++ {
			u[m.U.Index(x, y0-1, z)] = -u[m.U.Index(x, y0, z)]
			u[m.U.Index(x, y1, z)] = -u[m.U.Index(x, y1-1, z)]
		}
	}
	for y := y0 - 1; y <= y1; y++ {
		for x := ux0; x <= ux1; x++ {
			u[m.U.Index(x, y, z0-1)] = -u[m.U.Index(x, y, z0)]
			u[m.U.Index(x, y, z1)] = -u[m.U.Index(x, y, z1-1)]
		}
	}
	for z := z0; z < z1; z++ {
		for y := y0; y <= y1; y++ {
			v[m.V.Index(g-1, y, z)] = -v[m.V.Index(g, y, z)]
			v[m.V.Index(g+m.NX, y, z)] = -v[m.V.Index(g+m.NX-1, y, z)]
		}
	}
	for y := y0; y <= y1; y++ {
		for x := g - 1; x <= g+m.NX; x++ {
			v[m.V.Index(x, y, z0-1)] = -v[m.V.Index(x, y, z0)]
			v[m.V.Index(x, y, z1)] = -v[m.V.Index(x, y, z1-1)]
		}
	}
	for z := z0; z <= z1; z++ {
		for y := y0; y < y1; y++ {
			w[m.W.Index(g-1, y, z)] = -w[m.W.Index(g, y, z)]
			w[m.W.Index(g+m.NX, y, z)] = -w[m.W.Index(g+m.NX-1, y, z)]
		}
	}
	for z := z0; z <= z1; z++ {
		for x := g - 1; x <= g+m.NX; x++ {
			w[m.W.Index(x, y0-1, z)] = -w[m.W.Index(x, y0, z)]
			w[m.W.Index(x, y1, z)] = -w[m.W.Index(x, y1-1, z)]
		}
	}
}

// MaxVelocity returns the largest absolute velocity component, scanning in
// a fixed index order so the result is reproducible.
func (e *Engine) MaxVelocity() float64 {
	m := e.Mesh
	var mx float64
	for _, f := range []*Field{&m.U, &m.V, &m.W} {
		d := m.Data(f)
		for i := range d {
			if a := math.Abs(d[i]); a > mx {
				mx = a
			}
		}
	}
	return mx
}

// CFL returns the Courant number for the current field and timestep. Values
// above ~1 mean the semi-Lagrangian trace is crossing more than one cell
// per step: still stable, but increasingly diffusive.
func (e *Engine) CFL() float64 {
	return e.MaxVelocity() * e.Cfg.DT / e.Cfg.H
}
