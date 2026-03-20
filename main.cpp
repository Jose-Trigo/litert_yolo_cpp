#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>

#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " /path/to/model.tflite\n";
    return 1;
  }
  const char* model_path = argv[1];

  auto env_res = litert::Environment::Create({});
  if (!env_res) return 1;
  auto env = std::move(*env_res);

  auto opt_res = litert::Options::Create();
  if (!opt_res) return 1;
  auto options = std::move(*opt_res);

  auto model_res = litert::CompiledModel::Create(env, model_path, options);
  if (!model_res) {
    std::cerr << "Failed to load model: " << model_path << "\n";
    return 1;
  }
  auto model = std::move(*model_res);

  constexpr size_t kSignatureIndex = 0;

  auto in_res = model.CreateInputBuffers(kSignatureIndex);
  if (!in_res) return 1;
  auto inputs = std::move(*in_res);

  auto out_res = model.CreateOutputBuffers(kSignatureIndex);
  if (!out_res) return 1;
  auto outputs = std::move(*out_res);

  // Fill inputs with zeros by byte size (safe for any tensor type)
  for (auto& b : inputs) {
    auto sz_res = b.Size();
    if (!sz_res) return 1;
    std::vector<uint8_t> zeros(*sz_res, 0);
    if (!b.Write<uint8_t>(zeros)) return 1;
  }

  auto t0 = std::chrono::high_resolution_clock::now();
  auto run_status = model.Run(kSignatureIndex, inputs, outputs);
  auto t1 = std::chrono::high_resolution_clock::now();

  if (!run_status) {
    std::cerr << "Inference failed\n";
    return 1;
  }

  auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  std::cout << "Inference took " << us << " us\n";
  return 0;
}