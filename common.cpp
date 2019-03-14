#include "common.h"

#include <iostream>
#include <fstream>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

using std::string;
using std::vector;

namespace {

int reverse_int (int i)
{
  unsigned char c1, c2, c3, c4;

  c1 = i & 255;
  c2 = (i >> 8) & 255;
  c3 = (i >> 16) & 255;
  c4 = (i >> 24) & 255;

  return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

// Convert label info to one-hot format
vector<vector<float>> to_one_hot(vector<float> labels)
{
  vector<vector<float>> result;
  for (float label : labels) {
    vector<float> val(10);
    val[(int)label] = 1.0f;
    result.push_back(val);
  }
  return result;
}

// Create a Tensor(N, 784) from N image data
tensorflow::Tensor MakeTensor(const std::vector<vector<float>>& batch) {
  tensorflow::Tensor t(tensorflow::DT_FLOAT,
                       tensorflow::TensorShape({(int)batch.size(), 784}));
  auto dst = t.flat<float>().data();
  for (auto img : batch) {
    std::copy_n(img.begin(), 784, dst);
    dst += 784;
  }
  return t;
}

// Create a Tensor(N, 10) from N label data
tensorflow::Tensor MakeTargetTensor(const std::vector<vector<float>>& batch) {
  tensorflow::Tensor t(tensorflow::DT_FLOAT,
                       tensorflow::TensorShape({(int)batch.size(), 10}));
  auto dst = t.flat<float>().data();
  for (auto target : batch) {
    std::copy_n(target.begin(), 10, dst);
    dst += 10;
  }
  return t;
}

}


vector<vector<float>> read_training_file(string filename)
{
  std::ifstream ifs(filename.c_str(), std::ios::in | std::ios::binary);
  int magic_number = 0;
  int number_of_images = 0;
  int rows = 0;
  int cols = 0;

  // Read header
  ifs.read((char*)&magic_number, sizeof(magic_number));
  magic_number = reverse_int(magic_number);
  if (magic_number != 0x803) {
    std::cout << "Invalid magic number: " << magic_number << std::endl;
    std::cout << "Expected value is: " << 0x803 << std::endl;
    exit(1);
  }
  ifs.read((char*)&number_of_images, sizeof(number_of_images));
  number_of_images = reverse_int(number_of_images);
  ifs.read((char*)&rows, sizeof(rows));
  rows = reverse_int(rows);
  ifs.read((char*)&cols, sizeof(cols));
  cols = reverse_int(cols);

  vector<vector<float>> images(number_of_images);

  // Read image data
  for (int i = 0; i < number_of_images; i++) {
    images[i].resize(rows * cols);

    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < cols; col++) {
        unsigned char temp = 0;
        ifs.read((char*)&temp, sizeof(temp));
        images[i][rows*row+col] = (float)temp / 255.0f;
      }
    }
  }
  return images;
}

vector<float> read_label_file(string filename)
{
  std::ifstream ifs(filename.c_str(), std::ios::in | std::ios::binary);
  int magic_number = 0;
  int number_of_images = 0;

  // Read header
  ifs.read((char*)&magic_number, sizeof(magic_number));
  magic_number = reverse_int(magic_number);
  if (magic_number != 0x801) {
    std::cout << "Invalid magic number: " << magic_number << std::endl;
    std::cout << "Expected value is: " << 0x803 << std::endl;
    exit(1);
  }
  ifs.read((char*)&number_of_images, sizeof(number_of_images));
  number_of_images = reverse_int(number_of_images);

  vector<float> label(number_of_images);

  // Read image data
  for(int i = 0; i < number_of_images; i++){
    unsigned char temp = 0;
    ifs.read((char*)&temp, sizeof(temp));
    label[i] = (float)temp;
  }
  return label;
}

void predict(const std::unique_ptr<tensorflow::Session>& session, const vector<vector<float>>& batch, const vector<float>& labels) {
  // Create an input data
  tensorflow::Tensor lp(tensorflow::DT_BOOL, tensorflow::TensorShape({}));
  lp.flat<bool>().setZero();
  vector<std::pair<string, tensorflow::Tensor>> inputs = {
    {"input", MakeTensor(batch)},
    {"batch_normalization_1/keras_learning_phase", lp}
  };

  std::vector<tensorflow::Tensor> out_tensors;

  // Predict
  TF_CHECK_OK(session->Run(inputs, {"output/Softmax"}, {}, &out_tensors));

  // Calculate its accuracy
  int hits = 0;
  for (auto tensor : out_tensors) {
    auto items = tensor.shaped<float, 2>({static_cast<int>(batch.size()), 10});
    for (int i = 0; i < batch.size(); i++) {
      int arg_max = 0;
      float val_max = items(i, 0);
      for (int j = 0; j < 10; j++) {
        if (items(i, j) > val_max) {
          arg_max = j;
          val_max = items(i, j);
        }
      }
      if (arg_max == labels[i]) {
        hits++;
      }
    }
  }
  std::cout << "Accuracy: " << hits / (float)batch.size() << std::endl;
}

// Train
void run_train_step(const std::unique_ptr<tensorflow::Session>& session,
                    const std::vector<vector<float>>& train_x,
                    const std::vector<float>& train_y) {
  auto train_y_ = to_one_hot(train_y);
  vector<std::pair<string, tensorflow::Tensor>> inputs = {
    {"image", MakeTensor(train_x)},
    {"target", MakeTargetTensor(train_y_)}
  };
  TF_CHECK_OK(session->Run(inputs, {}, {"train"}, nullptr));
}
