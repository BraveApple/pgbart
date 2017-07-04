#!/usr/bin/env sh

cd ./src
make -f Makevars clean

cd ..
rm -f src/*.o
rm -f src/RcppExports.cpp
rm -f R/RcppExports.R
