# xproc package dev toolchains

all:
	@echo "Build necessary third party libraries"


test_TomoMorph:
	@echo "Testing converting tomo TIFF images to HDF5 archive."
	luigi --module xproc TomoMorph --conf examples/morph/morph_test.yml


format:
	@echo "evoke black to reforamt all source code"
	black `(pwd)
