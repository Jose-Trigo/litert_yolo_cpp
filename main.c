#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "litert/c/litert_environment.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_tensor_buffer.h"

int main() {
    LiteRtEnvironment env = NULL;
    LiteRtStatus status;

    // 1. Create environment
    status = LiteRtCreateEnvironment(&env);
    if (status != kLiteRtStatusOk) {
        printf("Failed to create environment\n");
        return 1;
    }

    // 2. Load model
    LiteRtCompiledModel model = NULL;
    status = LiteRtCompileModelFromFile(env, "model.tflite", &model);
    if (status != kLiteRtStatusOk) {
        printf("Failed to load model\n");
        return 1;
    }

    // 3. Create input buffer
    LiteRtTensorBuffer input = NULL;
    status = LiteRtCreateInputBuffer(model, 0, &input);
    if (status != kLiteRtStatusOk) {
        printf("Failed to create input buffer\n");
        return 1;
    }

    // 4. Create output buffer
    LiteRtTensorBuffer output = NULL;
    status = LiteRtCreateOutputBuffer(model, 0, &output);
    if (status != kLiteRtStatusOk) {
        printf("Failed to create output buffer\n");
        return 1;
    }

    // 5. Fill input with zeros
    size_t input_size = 0;
    LiteRtGetTensorBufferSize(input, &input_size);

    uint8_t* data = calloc(1, input_size);
    LiteRtWriteTensorBuffer(input, data, input_size);

    // 6. Run inference
    status = LiteRtRun(model, &input, &output);
    if (status != kLiteRtStatusOk) {
        printf("Inference failed\n");
        return 1;
    }

    // 7. Read output
    size_t output_size = 0;
    LiteRtGetTensorBufferSize(output, &output_size);

    uint8_t* out = malloc(output_size);
    LiteRtReadTensorBuffer(output, out, output_size);

    printf("Inference OK. First output byte = %d\n", out[0]);

    free(data);
    free(out);
    return 0;
}
