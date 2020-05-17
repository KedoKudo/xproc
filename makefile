# xproc package dev toolchains

all:
	@echo "Build necessary third party libraries"


format:
	@echo "evoke black to reforamt all source code"
	black `(pwd)
