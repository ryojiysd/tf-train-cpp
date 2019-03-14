//
// Example of training the model created by create_graph.py in a C++ program.
//

#include <iostream>
#include <vector>
#include <string>

#include "tensorflow/core/platform/init_main.h"

#include "common.h"

using std::vector;
using std::string;

namespace {

// Save a checkpoint
void save_checkpoint(const std::unique_ptr<tensorflow::Session>& session, const string& checkpoint_prefix) {
  tensorflow::Tensor ckpt(tensorflow::DT_STRING, tensorflow::TensorShape());
  ckpt.scalar<string>()() = checkpoint_prefix;
  TF_CHECK_OK(session->Run({{"save/Const", ckpt}}, {}, {"save/control_dependency"}, nullptr));
}

bool directory_exists(const string& dir) {
  struct stat buf;
  return stat(dir.c_str(), &buf) == 0;
}
}


int main(int argc, char* argv[]) {
  const string graph_def_filename = "model.pb";
  const string checkpoint_dir = "./checkpoints";
  const string checkpoint_prefix = checkpoint_dir + "/model.ckpt";

  // Setup global state for TensorFlow.
  tensorflow::port::InitMain(argv[0], &argc, &argv);

  std::cout << "Loading graph\n";
  tensorflow::GraphDef graph_def;
  // tensorflow::MetaGraphDef graph_def;
  TF_CHECK_OK(tensorflow::ReadBinaryProto(tensorflow::Env::Default(),
                                          graph_def_filename, &graph_def));

  std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_CHECK_OK(session->Create(graph_def));

  if (directory_exists(checkpoint_dir)) {
    std::cout << "Restoring model weights from checkpoint\n";
    tensorflow::Tensor ckpt(tensorflow::DT_STRING, tensorflow::TensorShape());
    ckpt.scalar<string>()() = checkpoint_prefix;
    TF_CHECK_OK(session->Run({{"save/Const", ckpt}}, {}, {"save/restore_all"}, nullptr));
  } else {
    std::cout << "Initializing model weights\n";
    TF_CHECK_OK(session->Run({}, {}, {"init"}, nullptr));
  }

  // Load images and labels of training data
  auto test_x = read_training_file("MNIST_data/t10k-images.idx3-ubyte");
  auto test_y = read_label_file("MNIST_data/t10k-labels.idx1-ubyte");
  predict(session, test_x, test_y);

  // Load images and labels of test data
  auto train_x = read_training_file("MNIST_data/train-images.idx3-ubyte");
  auto train_y = read_label_file("MNIST_data/train-labels.idx1-ubyte");

  // Training
  for (int i = 0; i < 20; ++i) {
    std::cout << "Epoch: " << i << std::endl;
    run_train_step(session, train_x, train_y);
  }

  std::cout << "Updated predictions\n";
  predict(session, test_x, test_y);

  std::cout << "Saving checkpoint\n";
  save_checkpoint(session, checkpoint_prefix);

  return 0;
}
