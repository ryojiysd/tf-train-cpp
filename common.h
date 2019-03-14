#pragma once

#include <string>
#include <vector>

#include "tensorflow/core/public/session.h"

std::vector<std::vector<float>> read_training_file(std::string filename);
std::vector<float> read_label_file(std::string filename);
void predict(const std::unique_ptr<tensorflow::Session>& session, const std::vector<std::vector<float>>& batch, const std::vector<float>& labels);
void run_train_step(const std::unique_ptr<tensorflow::Session>& session, const std::vector<std::vector<float>>& train_x, const std::vector<float>& train_y);
