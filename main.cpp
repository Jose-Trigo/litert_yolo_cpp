#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>

// C API
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"

static bool Ok(LiteRtStatus s) { return s == kLiteRtStatusOk; }

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " /path/to/model.tflite\n";
    return 1;
  }
  const char* model_path = argv[1];
  constexpr size_t kSignatureIndex = 0;

  LiteRtEnvironment env = nullptr;
  if (!Ok(LiteRtEnvironmentCreate(/*options=*/nullptr, /*num_options=*/0, &env))) {
    std::cerr << "LiteRtEnvironmentCreate failed\n";
    return 1;
  }

  LiteRtOptions options = nullptr;
  if (!Ok(LiteRtOptionsCreate(&options))) {
    std::cerr << "LiteRtOptionsCreate failed\n";
    LiteRtEnvironmentDestroy(env);
    return 1;
  }

  LiteRtCompiledModel model = nullptr;
  if (!Ok(LiteRtCompiledModelCreateFromFile(env, model_path, options, &model))) {
    std::cerr << "LiteRtCompiledModelCreateFromFile failed: " << model_path << "\n";
    LiteRtOptionsDestroy(options);
    LiteRtEnvironmentDestroy(env);
    return 1;
  }

  LiteRtTensorBufferArray inputs = nullptr;
  LiteRtTensorBufferArray outputs = nullptr;

  if (!Ok(LiteRtCompiledModelCreateInputBuffers(model, kSignatureIndex, &inputs)) ||
      !Ok(LiteRtCompiledModelCreateOutputBuffers(model, kSignatureIndex, &outputs))) {
    std::cerr << "CreateInputBuffers/CreateOutputBuffers failed\n";
    LiteRtCompiledModelDestroy(model);
    LiteRtOptionsDestroy(options);
    LiteRtEnvironmentDestroy(env);
    return 1;
  }

  // Zero-fill all input buffers by raw byte size
  size_t num_inputs = 0;
  if (!Ok(LiteRtTensorBufferArrayGetSize(inputs, &num_inputs))) {
    std::cerr << "LiteRtTensorBufferArrayGetSize failed\n";
    LiteRtTensorBufferArrayDestroy(outputs);
    LiteRtTensorBufferArrayDestroy(inputs);
    LiteRtCompiledModelDestroy(model);
    LiteRtOptionsDestroy(options);
    LiteRtEnvironmentDestroy(env);
    return 1;
  }

  for (size_t i = 0; i < num_inputs; ++i) {
    LiteRtTensorBuffer b = nullptr;
    size_t sz = 0;
    if (!Ok(LiteRtTensorBufferArrayGet(inputs, i, &b)) ||
        !Ok(LiteRtTensorBufferGetSize(b, &sz))) {
      std::cerr << "Input buffer query failed at index " << i << "\n";
      LiteRtTensorBufferArrayDestroy(outputs);
      LiteRtTensorBufferArrayDestroy(inputs);
      LiteRtCompiledModelDestroy(model);
      LiteRtOptionsDestroy(options);
      LiteRtEnvironmentDestroy(env);
      return 1;
    }

    std::vector<uint8_t> zeros(sz, 0);
    if (!Ok(LiteRtTensorBufferWrite(b, zeros.data(), sz))) {
      std::cerr << "Input write failed at index " << i << "\n";
      LiteRtTensorBufferArrayDestroy(outputs);
      LiteRtTensorBufferArrayDestroy(inputs);
      LiteRtCompiledModelDestroy(model);
      LiteRtOptionsDestroy(options);
      LiteRtEnvironmentDestroy(env);
      return 1;
    }
  }

  auto t0 = std::chrono::high_resolution_clock::now();
  LiteRtStatus run_status = LiteRtCompiledModelRun(model, kSignatureIndex, inputs, outputs);
  auto t1 = std::chrono::high_resolution_clock::now();

  if (!Ok(run_status)) {
    std::cerr << "Inference failed\n";
    LiteRtTensorBufferArrayDestroy(outputs);
    LiteRtTensorBufferArrayDestroy(inputs);
    LiteRtCompiledModelDestroy(model);
    LiteRtOptionsDestroy(options);
    LiteRtEnvironmentDestroy(env);
    return 1;
  }

  auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  std::cout << "Inference took " << us << " us\n";

  LiteRtTensorBufferArrayDestroy(outputs);
  LiteRtTensorBufferArrayDestroy(inputs);
  LiteRtCompiledModelDestroy(model);
  LiteRtOptionsDestroy(options);
  LiteRtEnvironmentDestroy(env);
  return 0;
}