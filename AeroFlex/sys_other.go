//go:build !darwin && !linux

package aeroflux

import (
	"errors"
	"runtime"
	"syscall"
)

// This file keeps the engine buildable on platforms without a hand-written
// mmap path. The arena degrades to a plain anonymous mapping via the stdlib
// wrapper, which means no guard pages and no mlock; NewArena still works,
// but the hardware containment guarantees do not apply.

const (
	protNone      = syscall.PROT_NONE
	protReadWrite = syscall.PROT_READ | syscall.PROT_WRITE
)

var errUnsupported = errors.New("aeroflux: raw memory syscalls unsupported on this platform")

func sysMmapAnonNone(length int) (uintptr, error)      { return 0, errUnsupported }
func sysMunmap(addr uintptr, length int) error         { return errUnsupported }
func sysMprotect(addr uintptr, length, prot int) error { return errUnsupported }
func sysMlock(addr uintptr, length int) error          { return errUnsupported }
func sysMunlock(addr uintptr, length int) error        { return errUnsupported }

func adviseHugePage(addr uintptr, length int) {}

// PinWorker locks the goroutine to its OS thread; no affinity API is wired
// up for this platform.
func PinWorker() error { runtime.LockOSThread(); return nil }

// UnpinWorker releases the OS-thread binding.
func UnpinWorker() { runtime.UnlockOSThread() }
