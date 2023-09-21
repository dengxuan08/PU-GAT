#!/usr/bin/env bash
system='linux'
tf_lib=$2
tf_inc=$3
cuda_lib=$4
cuda_inc=$5

if [ "$system" == "linux" ]; then
    g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I $tf_inc  -I $cuda_inc -I $tf_inc/external/nsync/public -lcudart -L $cuda_lib -L$tf_lib -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=1
elif [ "$system" == "centos" ]; then
    g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I $tf_inc  -I $cuda_inc -I $tf_inc/external/nsync/public -lcudart -L $cuda_lib -L$tf_lib -ltensorflow_framework -O2
else
    echo "unsupported system"
fi
