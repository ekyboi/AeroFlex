//go:build arm64

#include "textflag.h"

// func Relax()
//
// YIELD is the AArch64 spin-wait hint. On Apple Silicon it lets the core
// deprioritize this thread's front-end slots in favor of the sibling work
// that will actually release the lock, and lowers the power a hot spin
// draws.
TEXT ·Relax(SB), NOSPLIT|NOFRAME, $0-0
	YIELD
	RET
