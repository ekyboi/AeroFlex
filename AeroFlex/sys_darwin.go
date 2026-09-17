//go:build darwin

package aeroflux

import (
	"runtime"
	"syscall"
)

// Protection flags, aliased from stdlib syscall so arena.go stays platform-
// neutral and this file owns every platform constant.
const (
	protNone      = syscall.PROT_NONE
	protReadWrite = syscall.PROT_READ | syscall.PROT_WRITE
)

// sysMmapAnonNone reserves length bytes of anonymous, private, wholly
// inaccessible (PROT_NONE) address space and returns its base address.
//
// This goes through syscall.Syscall6 rather than syscall.Mmap because the
// stdlib wrapper returns a []byte whose backing pointer the runtime tracks;
// we want a bare uintptr the GC never sees, which is the entire basis of
// the heap-bypass guarantee.
func sysMmapAnonNone(length int) (uintptr, error) {
	addr, _, errno := syscall.Syscall6(
		syscall.SYS_MMAP,
		0, // let the kernel choose the address
		uintptr(length),
		protNone,
		syscall.MAP_PRIVATE|syscall.MAP_ANON,
		^uintptr(0), // fd = -1
		0,           // offset
	)
	if errno != 0 {
		return 0, errno
	}
	return addr, nil
}

func sysMunmap(addr uintptr, length int) error {
	if _, _, errno := syscall.Syscall(syscall.SYS_MUNMAP, addr, uintptr(length), 0); errno != 0 {
		return errno
	}
	return nil
}

func sysMprotect(addr uintptr, length, prot int) error {
	if _, _, errno := syscall.Syscall(syscall.SYS_MPROTECT, addr, uintptr(length), uintptr(prot)); errno != 0 {
		return errno
	}
	return nil
}

func sysMlock(addr uintptr, length int) error {
	if _, _, errno := syscall.Syscall(syscall.SYS_MLOCK, addr, uintptr(length), 0); errno != 0 {
		return errno
	}
	return nil
}

func sysMunlock(addr uintptr, length int) error {
	if _, _, errno := syscall.Syscall(syscall.SYS_MUNLOCK, addr, uintptr(length), 0); errno != 0 {
		return errno
	}
	return nil
}

// adviseHugePage is a no-op on Darwin.
//
// XNU exposes no MADV_HUGEPAGE equivalent: large-page promotion for
// anonymous memory is decided by the VM system, which already looks for
// exactly the shape NewArena produces (a large, 2 MiB-aligned anonymous
// mapping). There is no syscall to make the request explicit, so the
// alignment IS the hint. This exists to keep one call site in arena.go
// across platforms.
func adviseHugePage(addr uintptr, length int) {
	_, _ = addr, length
}

// PinWorker locks the calling goroutine to its OS thread and biases it away
// from the efficiency cores.
//
// Stated plainly, because it bears on what determinism claims are
// defensible: macOS does NOT expose per-core pinning to userspace. There is
// no sched_setaffinity equivalent. Two advisory mechanisms exist --
// THREAD_AFFINITY_POLICY (a cache-cluster grouping hint, reachable only
// through the Mach thread_policy_set trap, so it lives in the cgo file as
// SetClusterAffinity) and thread QoS/priority, handled here. Neither
// guarantees P-core residency.
//
// Therefore core placement is never load-bearing for numerical results in
// this engine. Determinism comes from the ticket turnstiles and a fixed
// reduction order, which hold regardless of where threads land.
//
// Clearing the background-QoS bit is the meaningful part: a thread left in
// PRIO_DARWIN_BG is strongly biased onto E-cores and throttled on I/O.
func PinWorker() error {
	runtime.LockOSThread()

	// PRIO_DARWIN_THREAD targets the calling thread. Priority 0 means "not
	// throttled"; PRIO_DARWIN_BG (0x1000) is the throttled state. This
	// clears any inherited background QoS.
	return syscall.Setpriority(prioDarwinThread, 0, 0)
}

// UnpinWorker releases the OS-thread binding taken by PinWorker.
func UnpinWorker() {
	runtime.UnlockOSThread()
}

// prioDarwinThread is PRIO_DARWIN_THREAD from <sys/resource.h>. It is not
// exported by the stdlib syscall package, so it is defined here.
const prioDarwinThread = 3
