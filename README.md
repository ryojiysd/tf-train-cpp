# TensorFlow training in C++

Create a DNN model for MNIST using Keras and train it using TensorFlow in C++.

## Requirement

- Python 3.6
- keras 2.2.4
- tensorflow 1.10.0
- Visual Studio 2015

## Setup

- Download MNIST Data from [here](http://yann.lecun.com/exdb/mnist/) and extract them into "MNIST_data" folder.
```
[MNIST_data]
  +- train-images.idx3-ubyte
  +- train-labels.idx1-ubyte
  +- t10k-images.idx3-ubyte
  +- t10k-labels.idx1-ubyte
```
- Download tensorflow 1.10.0 for C++ from [here](https://github.com/fo40225/tensorflow-windows-wheel) and unzip it into a directory.
- Set the path to the environment variable "TF_ROOT".
```
set TF_ROOT={PATH_TO_TENSOR_FLOW_DIR}\libtensorflow-cpu-windows-x86_64-1.10.0-avx2
```

- Copy "tensorflow.dll" to this directory.

## Create graph

- Install TensorFlow 1.10.0 and Keras.
- Execute the following command.

```
python create_graph.py
```

## Training

### Build C++ module

Open "Visual Studio Developper Command Prompt" and execute the following command.

```
cl /EHsc /I %TF_ROOT%/include \
  /I %TF_ROOT%/include/tensorflow/contrib/cmake/build \
  /I %TF_ROOT%/include/tensorflow/contrib/cmake/build/protobuf/src/protobuf/src \
  /I %TF_ROOT%/include/tensorflow/contrib/cmake/build/eigen/src/eigen \
  /I %TF_ROOT%/include/tensorflow/contrib/cmake/build/external/nsync/public \
  /D "COMPILER_MSVC" /D "WIN32" /D "NOMINMAX" /GS /GL /source-charset:utf-8 \
  train_and_predict.cpp common.cpp %TF_ROOT%/lib/tensorflow.lib
```

### Run

```
train_and_predict.exe
```

## Freeze Graph

```
python freeze_graph.py
```

## Inference

### Build C++ module

Open "Visual Studio Developper Command Prompt" and execute the following command.

```
cl /EHsc /I %TF_ROOT%/include \
  /I %TF_ROOT%/include/tensorflow/contrib/cmake/build \
  /I %TF_ROOT%/include/tensorflow/contrib/cmake/build/protobuf/src/protobuf/src \
  /I %TF_ROOT%/include/tensorflow/contrib/cmake/build/eigen/src/eigen \
  /I %TF_ROOT%/include/tensorflow/contrib/cmake/build/external/nsync/public \
  /D "COMPILER_MSVC" /D "WIN32" /D "NOMINMAX" /GS /GL /source-charset:utf-8 \
  predict.cpp common.cpp %TF_ROOT%/lib/tensorflow.lib
```

### Run

```
predict.exe
```

