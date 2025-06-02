// FreeRTOS and ESP includes
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "esp_timer.h"

// TFlite Micro includes
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "models/model.h"
#include "input/input_fomof_96.h"

// Camera module
//#include "static_image_provider.h"
//#include "camera_image_provider.h"


// Constants
const gpio_num_t g_led = GPIO_NUM_21;
tflite::MicroInterpreter* g_interpreter = nullptr;


bool setup() {
    // Setup LED
    gpio_set_direction(g_led, GPIO_MODE_OUTPUT);

    // Load Model
    // The model is defined as a C array, which we generated in the notebook
    const tflite::Model* model = ::tflite::GetModel(model_data);
    // if (model->version() != TFLITE_SCHEMA_VERSION) {
    //     printf(
    //         "Model provided is schema version %ld not equal to supported version %d.\n",
    //         model->version(),
    //         TFLITE_SCHEMA_VERSION
    //     );
    //     return false;
    // }

    // Operations Resolver
    // The model definition defines which operation to run and in what order, but this resolver links operations with the correct code to run.
    // TFLite Micro contains a resolver with all operations, but in order to limit code memory and compile time,
    // we can use a MutableOpResolver and specify which operations will be used.
    static tflite::MicroMutableOpResolver<6> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddAdd();
    // resolver.AddMean();
    resolver.AddMul();
    resolver.AddFullyConnected();
    resolver.AddReshape();
    

    // Allocate Memory
    // Instead of dynamically allocating memory, TFLite Micro works with a static memory pool.
    // All data necessary to run the model will be stored in here.
    // One optimization is to reduce the arena_size to get it as small as possible, whilst still being able to run the model.
    constexpr int tensor_arena_size = 3000 * 1024;
    EXT_RAM_BSS_ATTR static uint8_t tensor_arena[tensor_arena_size];
    // for SRAM: EXT_RAM_BSS_ATTR 

    // Interpreter
    // Finally we combine all the objects above into a single TFLite interpreter object.
    static tflite::MicroInterpreter interpreter(
        model,
        resolver,
        tensor_arena,
        tensor_arena_size
    );

    g_interpreter = &interpreter;

    // Allocate Tensors
    // Once the interpreter is build, we can try to allocate memory in our arena, to check whether we reserved enough memory.
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("Allocate tensors failed\n");
        return false;
    }

    return true;
}


void main_task(void)
{

    // Input & Output
    // These are the objects which we access in order to provide input and read the output from the model.
    TfLiteTensor* input = g_interpreter->input(0);
    TfLiteTensor* output = g_interpreter->output(0);

    memcpy(input->data.f, input_buffer, input_buffer_len * sizeof(float));

    // Run Model
    int64_t start_time = esp_timer_get_time();
    if (kTfLiteOk != g_interpreter->Invoke()) {
        printf("Invoke failed\n");
        return;
    }
    int64_t stop_time = esp_timer_get_time();
    
    // Print inference time
    printf("Count: %d | Inference Time: %.2f ms\n", output->data.int8[0], (stop_time - start_time) / 1000.0);

    while(1){
        vTaskDelay(1);
    }

}


// Main entry point of ESP
extern "C" void app_main()
{
    if (!setup()) {
        printf("Setup failed\n");
        while (true) {
            vTaskDelay(1);
        }
    }

    xTaskCreate((TaskFunction_t)&main_task, "main", 32 * 1024, NULL, 8, NULL);
    vTaskDelete(NULL);
}
