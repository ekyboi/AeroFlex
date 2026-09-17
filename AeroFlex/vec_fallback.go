//go:build !darwin || !cgo

package aeroflux

import (
	"math"
	"unsafe"
)

// Portable scalar implementations of the vector kernels Accelerate provides
// on Darwin. Same signatures and same arena-pointer discipline, so callers
// are identical across platforms; these are simply not vectorized by hand
// and rely on the compiler.

func vecSlice(p uintptr, n int) []float64 {
	return unsafe.Slice((*float64)(unsafe.Pointer(p)), n)
}

// VecSin computes dst[i] = sin(src[i]).
func VecSin(dst, src uintptr, n int) {
	if n <= 0 {
		return
	}
	d, s := vecSlice(dst, n), vecSlice(src, n)
	for i := range d {
		d[i] = math.Sin(s[i])
	}
}

// VecCos computes dst[i] = cos(src[i]).
func VecCos(dst, src uintptr, n int) {
	if n <= 0 {
		return
	}
	d, s := vecSlice(dst, n), vecSlice(src, n)
	for i := range d {
		d[i] = math.Cos(s[i])
	}
}

// VecExp computes dst[i] = exp(src[i]).
func VecExp(dst, src uintptr, n int) {
	if n <= 0 {
		return
	}
	d, s := vecSlice(dst, n), vecSlice(src, n)
	for i := range d {
		d[i] = math.Exp(s[i])
	}
}

// VecScaleAdd computes dst[i] = src[i]*scale + bias.
func VecScaleAdd(dst, src uintptr, scale, bias float64, n int) {
	if n <= 0 {
		return
	}
	d, s := vecSlice(dst, n), vecSlice(src, n)
	for i := range d {
		d[i] = s[i]*scale + bias
	}
}

// SetClusterAffinity is a no-op without the Mach thread-policy API.
func SetClusterAffinity(tag int32) error { _ = tag; return nil }
