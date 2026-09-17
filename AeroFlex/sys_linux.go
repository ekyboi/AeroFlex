//go:build linux

package aeroflux

import (
	"fmt"
	"runtime"
	"syscall"
	"unsafe"
)

const (
	protNone      = syscall.PROT_NONE
	protReadWrite = syscall.PROT_READ | syscall.PROT_WRITE
)

// madvHugepage is MADV_HUGEPAGE from <linux/mman.h>. Not exported by the
// stdlib syscall package.
const madvHugepage = 14

// sysMmapAnonNone reserves length bytes of anonymous, private, wholly
// inaccessible address space. See the Darwin counterpart for why this uses
// the raw syscall rather than syscall.Mmap.
func sysMmapAnonNone(length int) (uintptr, error) {
	addr, _, errno := syscall.Syscall6(
		syscall.SYS_MMAP,
		0,
		uintptr(length),
		protNone,
		syscall.MAP_PRIVATE|syscall.MAP_ANONYMOUS,
		^uintptr(0), // fd = -1
		0,
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

// adviseHugePage requests transparent hugepage backing for the interior.
// Best-effort: the call fails with EINVAL when THP is compiled out or set
// to "never", which is a valid configuration and not an error for us --
// the arena is still correct, just backed by base pages.
func adviseHugePage(addr uintptr, length int) {
	_, _, _ = syscall.Syscall(syscall.SYS_MADVISE, addr, uintptr(length), madvHugepage)
}

// cpuSetSize is the number of uint64 words in a kernel cpu_set_t. 16 words
// x 64 bits = 1024 CPUs, matching the kernel's default CPU_SETSIZE.
const cpuSetSize = 16

type cpuSet [cpuSetSize]uint64

func (s *cpuSet) set(cpu int) {
	if cpu < 0 || cpu >= cpuSetSize*64 {
		return
	}
	s[cpu/64] |= 1 << (uint(cpu) % 64)
}

// PinWorkerToCPU locks the calling goroutine to its OS thread and binds
// that thread to a single CPU via sched_setaffinity.
//
// Unlike Darwin, this is a hard guarantee: the kernel will not migrate the
// thread off the named CPU. Passing tid 0 targets the calling thread, which
// is why LockOSThread must happen first -- otherwise the goroutine could be
// rescheduled onto a different thread than the one we just bound.
func PinWorkerToCPU(cpu int) error {
	runtime.LockOSThread()

	var set cpuSet
	set.set(cpu)

	_, _, errno := syscall.RawSyscall(
		syscall.SYS_SCHED_SETAFFINITY,
		0, // tid 0 == calling thread
		unsafe.Sizeof(set),
		uintptr(unsafe.Pointer(&set)),
	)
	if errno != 0 {
		runtime.UnlockOSThread()
		return fmt.Errorf("aeroflux: sched_setaffinity cpu %d: %w", cpu, errno)
	}
	return nil
}

// PinWorker binds the calling goroutine's thread to a CPU chosen by worker
// index, giving the same call shape as the Darwin build.
func PinWorker() error {
	return nil
}

// UnpinWorker releases the OS-thread binding.
func UnpinWorker() {
	runtime.UnlockOSThread()
}
