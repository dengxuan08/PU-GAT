#!/usr/bin/env bash
# TF1.4
system='linux'
tf_lib=$2
tf_inc=$3
cuda_lib=$4
cuda_inc=$5


nvcc tf_approxmatch_g.cu -o tf_approxmatch_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
if [ "$system" == "linux" ]; then
    g++ -std=c++11 tf_approxmatch.cpp tf_approxmatch_g.cu.o -o tf_approxmatch_so.so -shared -fPIC -I $tf_inc  -I $cuda_inc -I $tf_inc/external/nsync/public -lcudart -L $cuda_lib -L$tf_lib -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=1
elif [ "$system" == "centos" ]; then
    g++ -std=c++11 tf_approxmatch.cpp tf_approxmatch_g.cu.o -o tf_approxmatch_so.so -shared -fPIC -I $tf_inc  -I $cuda_inc -I $tf_inc/external/nsync/public -lcudart -L $cuda_lib -L$tf_lib -ltensorflow_framework -O2
else
    echo "unsupported system"
fi

