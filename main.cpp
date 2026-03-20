#include <iostream>
#include <vector>
#include <chrono>
#include <filesystem>

#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_tensor_buffer.h"

int main() {
    const std::string model_path =
        "/home/efacec/ncnn_yolo_cpp/models/yolo_nano_v2_1_class_640_no_filter_int8.tflite";

    if (!std::filesystem::exists(model_path)) {
        std::cerr << "ERROR: Model not found: " << model_path << "\n";
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
        env, model_path, kLiteRtHwAcceleratorCpu);
    if (!model_result) {
        std::cerr << "ERROR: Failed to load model\n";
        return 1;
    }
    auto compiled_model = std::move(model_result.Value());

    // 3. Create input/output buffers
    auto in_bufs_result = compiled_model.CreateInputBuffers();
    auto out_bufs_result = compiled_model.CreateOutputBuffers();
    if (!in_bufs_result || !out_bufs_result) {
        std::cerr << "ERROR: Failed to create buffers\n";
        return 1;
    }

    auto input_buffers = std::move(in_bufs_result.Value());
    auto output_buffers = std::move(out_bufs_result.Value());

    // Fill input with zeros
    size_t input_size = input_buffers[0].SizeInBytes();
    std::vector<uint8_t> dummy(input_size, 0);

    input_buffers[0].Write<uint8_t>(
        absl::MakeConstSpan(dummy.data(), dummy.size())
    );

    // -------------------------------
    // Benchmark settings
    // -------------------------------
    const int warmup_runs = 10;
    const int timed_runs = 50;

    // Warmup
    for (int i = 0; i < warmup_runs; i++) {
        compiled_model.Run(input_buffers, output_buffers);
    }

    // Timed runs
    std::vector<double> times_ms;
    times_ms.reserve(timed_runs);

    for (int i = 0; i < timed_runs; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        compiled_model.Run(input_buffers, output_buffers);
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times_ms.push_back(ms);
    }

    // Compute stats
    double sum = 0, min_t = 1e9, max_t = 0;
    for (double t : times_ms) {
        sum += t;
        min_t = std::min(min_t, t);
        max_t = std::max(max_t, t);
    }
    double avg = sum / timed_runs;

    std::cout << "LiteRT Benchmark Results:\n";
    std::cout << "  Min: " << min_t << " ms\n";
    std::cout << "  Max: " << max_t << " ms\n";
    std::cout << "  Avg: " << avg << " ms\n";

    return 0;
}
