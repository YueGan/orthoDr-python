#!/bin/bash
rm ./python/cpp_exports.*.so
cmake .
make
cp cpp_exports.*.so ./test
mv cpp_exports.*.so ./python