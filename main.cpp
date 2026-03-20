#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>

#include "absl/types/span.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " /path/model.tflite [iters]\n";
    return 1;
  }

  const char* model_path = argv[1];
  const size_t iters = (argc >= 3) ? std::stoul(argv[2]) : 10;
  if (iters == 0) {
    std::cerr << "iters must be > 0\n";
    return 1;
  }

  constexpr size_t kSig = 0;

  auto env_or = litert::Environment::Create({});
  if (!env_or) {
    std::cerr << "Environment::Create failed\n";
    return 1;
  }
  auto env = std::move(env_or.Value());

  auto opts_or = litert::Options::Create();
  if (!opts_or) {
    std::cerr << "Options::Create failed\n";
    return 1;
  }
  auto opts = std::move(opts_or.Value());
  opts.SetHardwareAccelerators(litert::HwAccelerators::kCpu);

  auto model_or = litert::CompiledModel::Create(env, model_path, opts);
  if (!model_or) {
    std::cerr << "CompiledModel::Create failed\n";
    return 1;
  }
  auto model = std::move(model_or.Value());

  auto in_or = model.CreateInputBuffers(kSig);
  if (!in_or) {
    std::cerr << "CreateInputBuffers failed\n";
    return 1;
  }
  auto inputs = std::move(in_or.Value());

  auto out_or = model.CreateOutputBuffers(kSig);
  if (!out_or) {
    std::cerr << "CreateOutputBuffers failed\n";
    return 1;
  }
  auto outputs = std::move(out_or.Value());

  // Fill inputs with zeros
  for (auto& b : inputs) {
    auto sz_or = b.Size();
    if (!sz_or) {
      std::cerr << "TensorBuffer::Size failed\n";
      return 1;
    }
    std::vector<uint8_t> z(sz_or.Value(), 0);
    auto w = b.Write<uint8_t>(absl::MakeConstSpan(z));
    if (!w) {
      std::cerr << "TensorBuffer::Write failed\n";
      return 1;
    }
  }

  uint64_t total_us = 0;
  for (size_t i = 0; i < iters; ++i) {
    auto t0 = std::chrono::high_resolution_clock::now();
    auto run = model.Run(kSig, inputs, outputs);
    auto t1 = std::chrono::high_resolution_clock::now();
    if (!run) {
      std::cerr << "Run failed\n";
      return 1;
    }
    total_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  }

  std::cout << "avg_us=" << (total_us / iters) << "\n";
  return 0;
}