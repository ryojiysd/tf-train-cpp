//
// Example of training the model created by create_graph.py in a C++ program.
//

#include <iostream>
#include <string>

#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"

#include "common.h"

using std::string;

int main(int argc, char* argv[]) {
  const string graph_def_filename = "frozen_graph.pb";

  // Setup global state for TensorFlow.
  tensorflow::port::InitMain(argv[0], &argc, &argv);

  std::cout << "Loading graph\n";

  // Load a frozen model
  tensorflow::GraphDef graph_def;
  TF_CHECK_OK(tensorflow::ReadBinaryProto(tensorflow::Env::Default(),
                                          graph_def_filename, &graph_def));

  // Create a session
  std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_CHECK_OK(session->Create(graph_def));

  // Load images and labels of training data
  auto test_x = read_training_file("MNIST_data/t10k-images.idx3-ubyte");
  auto test_y = read_label_file("MNIST_data/t10k-labels.idx1-ubyte");

  predict(session, test_x, test_y);

  return 0;
}
