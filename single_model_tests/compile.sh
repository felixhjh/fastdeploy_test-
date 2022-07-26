#!/bin/bash

export CODE_PATH=/huangjianhui/fastdeploy_test/code
export http_proxy=http://172.19.56.199:3128
export https_proxy=http://172.19.56.199:3128
export fastdeploy_dir=${CODE_PATH}/FastDeploy
echo ${fastdeploy_dir}


rm -rf ${fastdeploy_dir}
if [ ! -d ${fastdeploy_dir} ]; then
    cd ${CODE_PATH}
    git clone https://github.com/PaddlePaddle/FastDeploy.git -b develop
    #git clone https://github.com/felixhjh/FastDeploy.git -b develop
fi

cd ${fastdeploy_dir}
git pull

function install() {
    export ENABLE_ORT_BACKEND=ON
    export ENABLE_PADDLE_BACKEND=ON
    export ENABLE_VISION=ON
    export WITH_GPU=ON
    export CUDA_DIRECTORY=/usr/local/cuda-11.2
    export TRT_DIRECTORY=/huangjianhui/fastdeploy_test/thirdparty/TensorRT-8.4.1.5
    export ENABLE_TRT_BACKEND=ON
    export ENABLE_PADDLE_FRONTEND=ON
    export ENABLE_DEBUG=OFF
    python setup.py build
    python setup.py bdist_wheel
    pip3 install --upgrade --no-deps --force-reinstall ./dist/*.whl
}

install

