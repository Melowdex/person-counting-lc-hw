#include "mbed.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "models/model.h"
#include "input/input_fomof_96.h"

Timer timer;
tflite::MicroInterpreter* g_interpreter = nullptr;


constexpr int kTensorArenaSize = 300 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

bool setup() {

    static tflite::MicroErrorReporter micro_error_reporter;
    const tflite::Model* model = ::tflite::GetModel(model_data);

    // Register required operators
    static tflite::MicroMutableOpResolver<6> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddAdd();
    // resolver.AddMean();   //-> only needed for GAP
    resolver.AddMul();      //-> only for mobilenet
    resolver.AddReshape();  //-> only for mobilenet
    resolver.AddFullyConnected();   //-> only for mobilenet

    // Create interpreter
    static tflite::MicroInterpreter interpreter(
        model,
        resolver,
        tensor_arena,
        kTensorArenaSize,
        &micro_error_reporter
    );

    g_interpreter = &interpreter;

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("Allocate tensors failed\n");
        return false;
    }

    printf("SUCCESS: tensors allocated successfully\n");
    return true;
}

void infer() {
    TfLiteTensor* input = g_interpreter->input(0);
    TfLiteTensor* output = g_interpreter->output(0);

    memcpy(input->data.f, input_buffer, input_buffer_len * sizeof(float));

    timer.start();
    if (kTfLiteOk != g_interpreter->Invoke()) {
        printf("Invoke failed\n");
        return;
    }
    timer.stop();

    float result = output->data.f[0];
    printf("Output type: %d\n", output->type);
    printf("Inference result = %d\n", output->data.int8[0]);
    printf("Time taken: %lld µs\n", timer.elapsed_time().count());
}

int main(void) {

    printf("Start setup\n");
    if (!setup()) {
        printf("Setup failed!\n");
        while (true);
    }

    printf("GOING TO INFER\n");
    infer();

    while (true) {}
}
