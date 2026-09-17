package aeroflux

import (
	"fmt"
	"unsafe"
)

// Field is a single scalar array resident in the arena, described by its
// padded extents and byte offset. X is the unit-stride axis: for fixed
// (y,z), consecutive x indices are adjacent float64s, which is what makes
// every sweep a linear walk the hardware prefetcher can follow.
type Field struct {
	// Off is the byte offset of element (0,0,0) within the arena.
	Off int
	// NX, NY, NZ are the padded extents including ghost cells.
	NX, NY, NZ int
	// SY, SZ are the element strides along Y and Z. SX is always 1.
	SY, SZ int
	// Count is the total padded element count.
	Count int
}

// Index returns the linear element index of padded coordinate (x,y,z).
func (f *Field) Index(x, y, z int) int {
	return x + y*f.SY + z*f.SZ
}

// ByteOffset returns the arena byte offset of padded coordinate (x,y,z).
func (f *Field) ByteOffset(x, y, z int) int {
	return f.Off + f.Index(x, y, z)*8
}

// MAC is a 3D Marker-and-Cell staggered grid.
//
// Staggering places pressure at cell centers and each velocity component on
// the face normal to its own axis:
//
//	p(i,j,k)   -> center of cell (i,j,k)
//	u(i,j,k)   -> face between cells (i-1,j,k) and (i,j,k)  [x-normal]
//	v(i,j,k)   -> face between cells (i,j-1,k) and (i,j,k)  [y-normal]
//	w(i,j,k)   -> face between cells (i,j,k-1) and (i,j,k)  [z-normal]
//
// The reason to stagger rather than collocate: on a collocated grid the
// second-difference pressure gradient decouples odd and even cells, letting
// a checkerboard pressure mode sit in the null space of the discrete
// operator and grow unchecked. With velocities on faces, the divergence and
// gradient operators are adjoint on a compact stencil and that spurious
// mode does not exist.
//
// A u-field therefore has one more cell along X than the pressure field,
// and likewise for v along Y and w along Z.
type MAC struct {
	// NX, NY, NZ are the interior (non-ghost) cell counts.
	NX, NY, NZ int
	// Ghost is the halo thickness on every face.
	Ghost int
	// H is the uniform cell size; DX==DY==DZ==H.
	H float64

	P    Field // pressure, cell-centered
	U    Field // x-velocity, x-faces
	V    Field // y-velocity, y-faces
	W    Field // z-velocity, z-faces
	Div  Field // velocity divergence / Poisson RHS, cell-centered
	Tmp0 Field // scratch, cell-centered sized (advection source)
	Tmp1 Field // scratch, cell-centered sized
	Tmp2 Field // scratch, cell-centered sized

	// Bytes is the total arena footprint of all fields.
	Bytes int

	arena *Arena
}

// fieldLayout lays out one field at the running offset, advancing it. Every
// field is padded up to a full 2 MiB boundary-friendly stride of
// CacheLineSize so that two fields never share a cache line and a worker
// writing the tail of one cannot invalidate the head of another.
func fieldLayout(off *int, nx, ny, nz int) Field {
	f := Field{
		Off:   *off,
		NX:    nx,
		NY:    ny,
		NZ:    nz,
		SY:    nx,
		SZ:    nx * ny,
		Count: nx * ny * nz,
	}
	sz := f.Count * 8
	// Round each field's footprint up to a cache line so the next field
	// starts on its own line.
	if r := sz % CacheLineSize; r != 0 {
		sz += CacheLineSize - r
	}
	*off += sz
	return f
}

// NewMAC lays out a staggered grid inside a, which must be large enough.
// Call MACBytes first to size the arena.
func NewMAC(a *Arena, nx, ny, nz, ghost int, h float64) (*MAC, error) {
	if nx < 1 || ny < 1 || nz < 1 {
		return nil, fmt.Errorf("aeroflux: grid extents must be >= 1, got %dx%dx%d", nx, ny, nz)
	}
	if ghost < 1 {
		return nil, fmt.Errorf("aeroflux: ghost thickness must be >= 1, got %d", ghost)
	}
	if h <= 0 {
		return nil, fmt.Errorf("aeroflux: cell size must be positive, got %v", h)
	}

	m := &MAC{NX: nx, NY: ny, NZ: nz, Ghost: ghost, H: h, arena: a}

	g2 := 2 * ghost
	// Padded cell-centered extents.
	cx, cy, cz := nx+g2, ny+g2, nz+g2

	off := 0
	m.P = fieldLayout(&off, cx, cy, cz)
	// Staggered components carry one extra plane on their own axis.
	m.U = fieldLayout(&off, cx+1, cy, cz)
	m.V = fieldLayout(&off, cx, cy+1, cz)
	m.W = fieldLayout(&off, cx, cy, cz+1)
	m.Div = fieldLayout(&off, cx, cy, cz)
	m.Tmp0 = fieldLayout(&off, cx+1, cy, cz)
	m.Tmp1 = fieldLayout(&off, cx, cy+1, cz)
	m.Tmp2 = fieldLayout(&off, cx, cy, cz+1)
	m.Bytes = off

	if a != nil && off > a.Len {
		return nil, fmt.Errorf("aeroflux: grid needs %d bytes, arena has %d", off, a.Len)
	}
	return m, nil
}

// MACBytes reports the arena footprint a grid of these dimensions needs,
// so callers can size the arena before allocating it.
func MACBytes(nx, ny, nz, ghost int) int {
	m, err := NewMAC(nil, nx, ny, nz, ghost, 1)
	if err != nil {
		return 0
	}
	return m.Bytes
}

// Data returns the float64 view of field f.
func (m *MAC) Data(f *Field) []float64 {
	return m.arena.Float64s(f.Off, f.Count)
}

// Ptr returns the raw arena address of element (x,y,z) of field f, for
// handing to Accelerate.
func (m *MAC) Ptr(f *Field, x, y, z int) uintptr {
	return m.arena.Pointer(f.ByteOffset(x, y, z))
}

// Zero clears every field. Setup-only: it touches the whole arena.
func (m *MAC) Zero() {
	b := m.arena.Bytes()
	for i := range b[:m.Bytes] {
		b[i] = 0
	}
}

// InteriorX returns the half-open padded x-range of interior cells for a
// cell-centered field.
func (m *MAC) InteriorX() (lo, hi int) { return m.Ghost, m.Ghost + m.NX }

// InteriorY returns the half-open padded y-range of interior cells.
func (m *MAC) InteriorY() (lo, hi int) { return m.Ghost, m.Ghost + m.NY }

// InteriorZ returns the half-open padded z-range of interior cells.
func (m *MAC) InteriorZ() (lo, hi int) { return m.Ghost, m.Ghost + m.NZ }

// CellCount returns the number of interior cells, which is the order of the
// pressure Poisson system.
func (m *MAC) CellCount() int { return m.NX * m.NY * m.NZ }

// CellIndex maps interior cell coordinates (0-based, excluding ghosts) to
// the Poisson system's row/column index. X varies fastest to match the
// field memory order, so the matrix bandwidth follows the same axis the
// sweeps do.
func (m *MAC) CellIndex(i, j, k int) int {
	return i + j*m.NX + k*m.NX*m.NY
}

// Compile-time check that Field's hot accessor fields stay 8-byte aligned
// and the struct does not accidentally grow padding that would push a
// second Field onto a shared line when stored in an array.
const (
	_ = uint(unsafe.Alignof(Field{})) - 8
	_ = uint(unsafe.Offsetof(Field{}.NX)) - 8
)
