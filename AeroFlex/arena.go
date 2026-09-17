package aeroflux

import (
	"fmt"
	"os"
	"runtime/debug"
	"sync/atomic"
	"unsafe"
)

// PageSize is resolved once at process start from the OS (16 KiB on Apple
// Silicon Darwin, typically 4 KiB on Linux). All alignment math in this
// file uses this runtime value rather than a hardcoded constant: hardcoding
// 4096 would under-size every guard page on Darwin, leaving the "guard"
// only 1/4 covering its intended page and letting a stencil overread land
// in live memory instead of faulting.
var PageSize = os.Getpagesize()

// HugePageSize is the alignment target for the interior data region. 2 MiB
// matches both x86-64 and AArch64 large-page granularity. The interior is
// rounded up to and aligned on this boundary so the kernel's VM can back it
// with large pages, cutting TLB pressure across the full-grid sweeps.
const HugePageSize = 2 * 1024 * 1024

func init() {
	// Convert memory-access faults into recoverable Go panics. A stencil
	// overread that lands in a PROT_NONE guard page raises SIGSEGV; with
	// this set, the runtime delivers it as a panic on the faulting
	// goroutine instead of killing the process.
	//
	// This does NOT by itself contain the blast radius: an unrecovered
	// panic still takes down the whole program. Containment requires each
	// worker goroutine to run its hot loop under a deferred recover --
	// see RecoverFault. SetPanicOnFault is per-goroutine state, so every
	// worker must also set it for itself at startup; calling it here only
	// covers the goroutine that imports/initializes the package.
	debug.SetPanicOnFault(true)
}

// Arena is a raw, off-heap region backed by an anonymous mmap mapping,
// optionally mlock'd against paging, and flanked by PROT_NONE guard pages.
//
// Base is deliberately a uintptr rather than an unsafe.Pointer: the garbage
// collector does not trace uintptr, which is precisely the property that
// keeps this memory invisible to the Go heap. The consequence is that the
// arena's lifetime is entirely manual -- nothing keeps it alive but the
// Arena value itself, and use-after-Close is undefined behavior that will
// typically fault rather than read stale data (the mapping is gone).
type Arena struct {
	// mapBase/mapLen describe the final reservation after trimming,
	// including both guard pages. munmap needs these verbatim.
	mapBase uintptr
	mapLen  int

	// Base is the first usable byte (PROT_READ|PROT_WRITE), immediately
	// after the leading guard page, aligned to HugePageSize. Len is the
	// usable size in bytes, rounded up to HugePageSize.
	Base uintptr
	Len  int

	locked atomic.Bool
	closed atomic.Bool
}

// NewArena reserves a guarded, hugepage-aligned off-heap region of at least
// size usable bytes.
//
// The final mapping is laid out as:
//
//	[ guard page ][ usable region, HugePageSize-aligned ][ guard page ]
//
// Over-allocate-and-trim: we reserve enough slack to guarantee a
// HugePageSize-aligned interior exists somewhere inside the reservation,
// locate it, then munmap the excess on both ends so the alignment slack is
// returned to the OS rather than squatting on address space.
func NewArena(size int) (*Arena, error) {
	if size <= 0 {
		return nil, fmt.Errorf("aeroflux: arena size must be positive, got %d", size)
	}

	usableLen := roundUp(size, HugePageSize)

	// Worst case the aligned interior starts HugePageSize-1 bytes into the
	// reservation, and we need a full page of guard on each side.
	reserveLen := usableLen + 2*PageSize + HugePageSize

	// Reserve as PROT_NONE. Everything starts inaccessible; we open up
	// only the interior below. That ordering means there is never a window
	// where the guard pages are readable.
	mapBase, err := sysMmapAnonNone(reserveLen)
	if err != nil {
		return nil, fmt.Errorf("aeroflux: mmap reserve %d bytes: %w", reserveLen, err)
	}

	// Carve the aligned interior out of the middle, leaving at least one
	// page of headroom in front for the leading guard.
	alignedStart := roundUpPtr(mapBase+uintptr(PageSize), HugePageSize)
	guardFront := alignedStart - uintptr(PageSize)
	usableEnd := alignedStart + uintptr(usableLen)
	guardBack := usableEnd

	frontTrim := int(guardFront - mapBase)
	backTrim := int((mapBase + uintptr(reserveLen)) - (guardBack + uintptr(PageSize)))

	if frontTrim > 0 {
		if err := sysMunmap(mapBase, frontTrim); err != nil {
			_ = sysMunmap(mapBase, reserveLen)
			return nil, fmt.Errorf("aeroflux: munmap front trim: %w", err)
		}
	}
	if backTrim > 0 {
		if err := sysMunmap(guardBack+uintptr(PageSize), backTrim); err != nil {
			_ = sysMunmap(guardFront, int(guardBack+uintptr(PageSize)-guardFront))
			return nil, fmt.Errorf("aeroflux: munmap back trim: %w", err)
		}
	}

	a := &Arena{
		mapBase: guardFront,
		mapLen:  int(guardBack+uintptr(PageSize)) - int(guardFront),
		Base:    alignedStart,
		Len:     usableLen,
	}

	// Open up only the interior. The two guard pages keep the PROT_NONE
	// they were born with, so any access into them faults with no branch
	// in the hot path.
	if err := sysMprotect(alignedStart, usableLen, protReadWrite); err != nil {
		_ = a.release()
		return nil, fmt.Errorf("aeroflux: mprotect interior rw: %w", err)
	}

	// mlock is best-effort: it commonly fails on RLIMIT_MEMLOCK without
	// privilege. The arena is still correct when unlocked, just pageable,
	// which costs tail latency rather than correctness. Callers needing
	// hard residency guarantees must check Locked().
	if err := sysMlock(alignedStart, usableLen); err == nil {
		a.locked.Store(true)
	}

	adviseHugePage(alignedStart, usableLen)

	return a, nil
}

// Locked reports whether the usable region is currently resident via mlock.
func (a *Arena) Locked() bool { return a.locked.Load() }

// Pointer returns the raw address at byte offset off within the usable
// region. It returns uintptr rather than unsafe.Pointer so that callers
// must perform an explicit conversion, making each crossing into unsafe
// territory visible at the call site.
//
// The bounds check here is a programming-error guard for setup code. It is
// not a substitute for the guard pages: hot-path stencil code indexes the
// arena directly without going through Pointer.
func (a *Arena) Pointer(off int) uintptr {
	if off < 0 || off > a.Len {
		panic(fmt.Sprintf("aeroflux: arena offset %d out of range [0,%d]", off, a.Len))
	}
	return a.Base + uintptr(off)
}

// Float64s returns a []float64 view over count elements starting at byte
// offset off. The returned slice aliases arena memory directly: it is not
// heap-allocated, writing through it writes the arena, and it must not
// outlive Close.
//
// go vet reports "possible misuse of unsafe.Pointer" on the conversion
// below, and the warning is expected rather than a defect to be silenced.
// Vet's rule assumes a uintptr may hold a stale address of a GC-managed
// object that could be moved or freed between the integer and pointer
// forms. That hazard does not exist here: the address refers to an
// anonymous mmap region the Go collector neither traces, moves, nor
// reclaims, and whose lifetime ends only at Close. Converting through
// uintptr is precisely how the arena stays invisible to the GC. Do not
// "fix" this by storing Base as an unsafe.Pointer -- that would make the
// runtime treat arena addresses as live heap references and defeat the
// zero-heap guarantee the engine is built on.
func (a *Arena) Float64s(off, count int) []float64 {
	const esz = int(unsafe.Sizeof(float64(0)))
	if off < 0 || count < 0 || off+count*esz > a.Len {
		panic(fmt.Sprintf("aeroflux: float64 view [%d,+%d) out of range [0,%d]", off, count*esz, a.Len))
	}
	if (a.Base+uintptr(off))%uintptr(esz) != 0 {
		panic(fmt.Sprintf("aeroflux: float64 view at offset %d is not 8-byte aligned", off))
	}
	return unsafe.Slice((*float64)(unsafe.Pointer(a.Base+uintptr(off))), count)
}

// Bytes returns a []byte view over the full usable region. Same aliasing,
// lifetime, and unsafe.Pointer rules as Float64s.
func (a *Arena) Bytes() []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(a.Base)), a.Len)
}

// Close unlocks and unmaps the entire reservation, guard pages included.
// It is idempotent. Close must not race with any in-flight access to arena
// memory; after it returns, every previously handed-out view is dangling
// and touching one will fault.
func (a *Arena) Close() error {
	if !a.closed.CompareAndSwap(false, true) {
		return nil
	}
	return a.release()
}

func (a *Arena) release() error {
	if a.locked.CompareAndSwap(true, false) {
		_ = sysMunlock(a.Base, a.Len)
	}
	return sysMunmap(a.mapBase, a.mapLen)
}

// RecoverFault converts a guard-page fault (delivered as a runtime panic
// because of SetPanicOnFault) into an error, and re-panics anything that is
// not a fault. Worker goroutines defer this so a blown-up cell -- a
// finite-time singularity driving an index past the grid, for instance --
// takes down that worker's step rather than the process.
//
// Every worker goroutine must call debug.SetPanicOnFault(true) itself:
// the setting is per-goroutine, and the package init only covers the
// initializing one.
//
// The recovered value for a fault is a runtime.Error whose concrete type is
// *runtime.panicmemError. Matching on the interface is deliberate: anything
// that is not a runtime.Error is a real bug (nil map write, bad type
// assertion) and must keep propagating rather than being silently demoted
// to an error return.
func RecoverFault(dst *error) {
	r := recover()
	if r == nil {
		return
	}
	if re, ok := r.(interface {
		error
		RuntimeError()
	}); ok {
		*dst = fmt.Errorf("aeroflux: guard-page fault contained: %w", re)
		return
	}
	panic(r)
}

func roundUp(n, align int) int {
	if align <= 0 {
		return n
	}
	if rem := n % align; rem != 0 {
		return n + (align - rem)
	}
	return n
}

func roundUpPtr(p, align uintptr) uintptr {
	if align == 0 {
		return p
	}
	if rem := p % align; rem != 0 {
		return p + (align - rem)
	}
	return p
}
