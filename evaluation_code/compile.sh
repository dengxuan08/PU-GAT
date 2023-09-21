#/usr/bin/env bash
# sudo apt-get install libcgal-dev
#or module load cgal
cmake .
make
./evaluation camel.off camel.xyz
./evaluation chair.off chair.xyz
./evaluation cow.off cow.xyz
./evaluation duck.off duck.xyz
./evaluation elephant.off elephant.xyz
./evaluation horse.off horse.xyz
./evaluation star.off star.xyz
#./evaluation tiger.off tiger.xyz