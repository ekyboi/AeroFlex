//go:build amd64

#include "textflag.h"

// func Relax()
//
// PAUSE hints to the x86 pipeline that this is a spin-wait, avoiding the
// memory-order-violation pipeline flush on loop exit and reducing power.
TEXT ·Relax(SB), NOSPLIT|NOFRAME, $0-0
	PAUSE
	RET
