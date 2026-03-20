#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_tensor_buffer.h"

int main() {
  // 1. Create environment (no options)
  auto env_result = litert::Environment::Create({});
  if (!env_result) {
    std::cerr << "Failed to create environment\n";
    return 1;
  }
  auto env = std::move(*env_result);

  // 2. Create options (defaults: CPU)
  auto options_result = litert::Options::Create();
  if (!options_result) {
    std::cerr << "Failed to create options\n";
    return 1;
  }
  auto options = std::move(*options_result);

  // 3. Load model (replace with your model path)
  const char* model_path = "your_model.tflite";
  auto model_result = litert::CompiledModel::Create(env, model_path, options);
  if (!model_result) {
    std::cerr << "Failed to load model\n";
    return 1;
  }
  auto model = std::move(*model_result);

  // 4. Create input/output buffers (signature 0)
  auto in_result = model.CreateInputBuffers(0);
  if (!in_result) {
    std::cerr << "Failed to create input buffers\n";
    return 1;
  }
  auto input_buffers = std::move(*in_result);

  auto out_result = model.CreateOutputBuffers(0);
  if (!out_result) {
    std::cerr << "Failed to create output buffers\n";
    return 1;
  }
  auto output_buffers = std::move(*out_result);

  // 5. Fill input buffers with zeros (or your data)
  for (auto& buf : input_buffers) {
    auto type_result = buf.TensorType();
    if (!type_result) continue;
    auto type = *type_result;
    size_t total = 1;
    for (size_t d = 0; d < type.Layout().Rank(); ++d)
      total *= type.Layout().Dimensions()[d];
    std::vector<uint8_t> zeros(total * 4, 0); // 4 bytes per element (float/int32)
    buf.Write<uint8_t>(absl::MakeConstSpan(zeros));
  }

  // 6. Run inference and time it
  auto start = std::chrono::high_resolution_clock::now();
  auto status = model.Run(0, input_buffers, output_buffers);
  auto end = std::chrono::high_resolution_clock::now();
  if (!status) {
    std::cerr << "Inference failed\n";
    return 1;
  }
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "Inference took " << duration.count() << " us\n";

  return 0;
}