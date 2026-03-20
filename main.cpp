// minimal_litert_run.cpp
// A tiny, adapted version of run_model.cc core logic.

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"

namespace {

using litert::CompiledModel;
using litert::Environment;
using litert::Expected;
using litert::TensorBuffer;

// Run a single inference on signature 0.
Expected<void> RunOnce(absl::string_view model_path) {
  // 1. Create environment (no special options).
  LITERT_ASSIGN_OR_RETURN(auto env, Environment::Create({}));

  // 2. Create compiled model on CPU.
  //    This matches the docs: CompiledModel::Create(env, "mymodel.tflite", kLiteRtHwAcceleratorCpu)
  LITERT_ASSIGN_OR_RETURN(
      auto compiled_model,
      CompiledModel::Create(env, std::string(model_path),
                            litert::HwAccelerators::kCpu));

  const size_t kSignatureIndex = 0;

  // 3. Create input/output buffers for that signature.
  LITERT_ASSIGN_OR_RETURN(auto input_buffers,
                          compiled_model.CreateInputBuffers(kSignatureIndex));
  LITERT_ASSIGN_OR_RETURN(auto output_buffers,
                          compiled_model.CreateOutputBuffers(kSignatureIndex));

  if (input_buffers.empty() || output_buffers.empty()) {
    return litert::Error(litert::Status::kErrorInvalidArgument,
                         "Model has no inputs or outputs.");
  }

  // 4. Fill first input with zeros.
  //    NOTE: TensorBuffer::Size() returns size in bytes.
  LITERT_ASSIGN_OR_RETURN(size_t input_size_bytes,
                          input_buffers[0].Size());
  std::vector<uint8_t> input_data(input_size_bytes, 0);

  LITERT_RETURN_IF_ERROR(
      input_buffers[0].Write<uint8_t>(absl::MakeConstSpan(input_data)));

  // 5. Run the model once.
  LITERT_RETURN_IF_ERROR(
      compiled_model.Run(kSignatureIndex, input_buffers, output_buffers));

  // 6. Read back first output (just to prove it works).
  LITERT_ASSIGN_OR_RETURN(size_t output_size_bytes,
                          output_buffers[0].Size());
  std::vector<uint8_t> output_data(output_size_bytes);

  LITERT_RETURN_IF_ERROR(
      output_buffers[0].Read<uint8_t>(absl::MakeSpan(output_data)));

  // Print a few bytes.
  std::cout << "Output[0] first bytes: ";
  for (size_t i = 0; i < std::min<size_t>(10, output_data.size()); ++i) {
    std::cout << static_cast<int>(output_data[i]) << " ";
  }
  std::cout << std::endl;

  return {};
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: litert_bench <model.tflite>\n";
    return EXIT_FAILURE;
  }

  absl::string_view model_path(argv[1]);
  auto res = RunOnce(model_path);
  if (!res) {
    ABSL_LOG(ERROR) << "Error: " << res.Error().Message();
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
