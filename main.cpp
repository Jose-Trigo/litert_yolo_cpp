#include <iostream>
#include <vector>
#include <filesystem>

#include "litert_environment.h"
#include "litert_compiled_model.h"
#include "litert_tensor_buffer.h"

int main() {
    // Check model exists
    if (!std::filesystem::exists("model.tflite")) {
        std::cerr << "ERROR: model.tflite not found in working directory\n";
        return 1;
    }

    // 1. Create environment
    auto env_result = litert::Environment::Create({});
    if (!env_result) {
        std::cerr << "ERROR: Failed to create Environment\n";
        return 1;
    }
    auto env = std::move(env_result.Value());

    // 2. Load model (CPU)
    auto model_result = litert::CompiledModel::Create(
        env,
        "model.tflite",
        kLiteRtHwAcceleratorCpu
    );
    if (!model_result) {
        std::cerr << "ERROR: Failed to load model\n";
        return 1;
    }
    auto compiled_model = std::move(model_result.Value());

    // 3. Create input buffers
    auto in_bufs_result = compiled_model.CreateInputBuffers();
    if (!in_bufs_result) {
        std::cerr << "ERROR: Failed to create input buffers\n";
        return 1;
    }
    auto input_buffers = std::move(in_bufs_result.Value());

    // 4. Create output buffers
    auto out_bufs_result = compiled_model.CreateOutputBuffers();
    if (!out_bufs_result) {
        std::cerr << "ERROR: Failed to create output buffers\n";
        return 1;
    }
    auto output_buffers = std::move(out_bufs_result.Value());

    // 5. Fill input with zeros
    size_t input_size = input_buffers[0].SizeInBytes();
    std::vector<uint8_t> dummy(input_size, 0);

    auto write_ok = input_buffers[0].Write<uint8_t>(absl::MakeConstSpan(dummy));
    if (!write_ok) {
        std::cerr << "ERROR: Failed to write input buffer\n";
        return 1;
    }

    // 6. Run inference
    auto run_status = compiled_model.Run(input_buffers, output_buffers);
    if (!run_status.ok()) {
        std::cerr << "ERROR: Model Run() failed\n";
        return 1;
    }

    // 7. Read output
    size_t out_size = output_buffers[0].SizeInBytes();
    std::vector<uint8_t> out(out_size);

    auto read_ok = output_buffers[0].Read<uint8_t>(absl::MakeSpan(out));
    if (!read_ok) {
        std::cerr << "ERROR: Failed to read output buffer\n";
        return 1;
    }

    // 8. Print first output value
    std::cout << "Inference OK. Output[0] = " << (int)out[0] << "\n";

    return 0;
}
