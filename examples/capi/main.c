#include <rustnn/rustnn.h>

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int check(RustnnStatus status) {
  if (status == RustnnStatus_Success) {
    return 1;
  }
  const char *message = rustnn_last_error_message();
  fprintf(stderr, "rustnn error: %s\n", message != NULL ? message : "unknown error");
  return 0;
}

int main(void) {
  rustnn_init_logger();

  RustnnContext *context = NULL;
  RustnnGraphBuilder *builder = NULL;
  RustnnGraph *graph = NULL;
  RustnnOperandDescriptor *matrix = NULL;
  RustnnTensorDescriptor *input_tensor_descriptor = NULL;
  RustnnTensorDescriptor *output_tensor_descriptor = NULL;
  RustnnTensor *input_tensor = NULL;
  RustnnTensor *output_tensor = NULL;
  RustnnOperand *input = NULL;
  RustnnOperand *bias = NULL;
  RustnnOperand *scale = NULL;
  RustnnOperand *shifted = NULL;
  RustnnOperand *scaled = NULL;
  RustnnOperand *output = NULL;
  char *webnn_text = NULL;
  int result = EXIT_FAILURE;

  const uint64_t shape[] = {2, 2};
  const float bias_values[] = {1.0f, 2.0f, 3.0f, 4.0f};
  const float scale_values[] = {2.0f, 2.0f, 2.0f, 2.0f};
  const RustnnOperatorOptions add_options = {"add bias"};
  const RustnnOperatorOptions multiply_options = {"scale values"};
  const RustnnOperatorOptions relu_options = {"clamp negatives"};
  RustnnContextOptions context_options = rustnn_context_options_default();
  context_options.backend_hint = RustnnBackend_Automatic;

  if (!check(rustnn_context_create(&context_options, &context)) ||
      !check(rustnn_graph_builder_create(context, &builder)) ||
      !check(rustnn_operand_descriptor_create(RustnnDataType_Float32, shape, 2, &matrix)) ||
      !check(rustnn_graph_builder_input(builder, "input", matrix, &input)) ||
      !check(rustnn_graph_builder_constant(builder, matrix, bias_values,
                                           sizeof(bias_values), &bias)) ||
      !check(rustnn_graph_builder_constant(builder, matrix, scale_values,
                                           sizeof(scale_values), &scale)) ||
      !check(rustnn_graph_builder_add_with_options(builder, input, bias, &add_options,
                                                   &shifted)) ||
      !check(rustnn_graph_builder_multiply_with_options(
          builder, shifted, scale, &multiply_options, &scaled)) ||
      !check(rustnn_graph_builder_relu_with_options(builder, scaled, &relu_options,
                                                    &output))) {
    goto cleanup;
  }

  size_t rank = 0;
  if (!check(rustnn_graph_builder_operand_shape(builder, output, NULL, 0, &rank))) {
    goto cleanup;
  }
  uint64_t *output_shape = calloc(rank, sizeof(*output_shape));
  if (rank != 0 && output_shape == NULL) {
    fprintf(stderr, "allocation failed\n");
    goto cleanup;
  }
  if (!check(rustnn_graph_builder_operand_shape(builder, output, output_shape, rank,
                                                &rank))) {
    free(output_shape);
    goto cleanup;
  }
  printf("output shape:");
  for (size_t index = 0; index < rank; ++index) {
    printf(" %" PRIu64, output_shape[index]);
  }
  free(output_shape);
  printf("\n\n");

  const RustnnNamedOperand outputs[] = {{"output", output}};
  if (!check(rustnn_graph_builder_webnn_text(builder, outputs, 1, &webnn_text))) {
    goto cleanup;
  }
  puts(webnn_text);

  if (!check(rustnn_graph_builder_build(&builder, outputs, 1, &graph)) ||
      !check(rustnn_tensor_descriptor_create(RustnnDataType_Float32, shape, 2, false,
                                             true, &input_tensor_descriptor)) ||
      !check(rustnn_tensor_descriptor_create(RustnnDataType_Float32, shape, 2, true,
                                             false, &output_tensor_descriptor)) ||
      !check(rustnn_context_create_tensor(context, input_tensor_descriptor,
                                          &input_tensor)) ||
      !check(rustnn_context_create_tensor(context, output_tensor_descriptor,
                                          &output_tensor))) {
    goto cleanup;
  }

  const float input_values[] = {1.0f, -3.0f, 2.0f, -10.0f};
  float output_values[4] = {0};
  const float expected_values[] = {4.0f, 0.0f, 10.0f, 0.0f};
  const RustnnNamedTensor inputs[] = {{"input", input_tensor}};
  const RustnnNamedTensor tensor_outputs[] = {{"output", output_tensor}};
  if (!check(rustnn_context_write_tensor(context, input_tensor, input_values,
                                         sizeof(input_values))) ||
      !check(rustnn_context_dispatch(context, graph, inputs, 1, tensor_outputs, 1)) ||
      !check(rustnn_context_read_tensor(context, output_tensor, output_values,
                                        sizeof(output_values)))) {
    goto cleanup;
  }
  for (size_t index = 0; index < 4; ++index) {
    if (output_values[index] != expected_values[index]) {
      fprintf(stderr, "unexpected output value at index %zu\n", index);
      goto cleanup;
    }
  }
  printf("dispatch output:");
  for (size_t index = 0; index < 4; ++index) {
    printf(" %g", output_values[index]);
  }
  printf("\n");
  result = EXIT_SUCCESS;

cleanup:
  rustnn_string_destroy(webnn_text);
  rustnn_tensor_destroy(output_tensor);
  rustnn_tensor_destroy(input_tensor);
  rustnn_tensor_descriptor_destroy(output_tensor_descriptor);
  rustnn_tensor_descriptor_destroy(input_tensor_descriptor);
  rustnn_graph_destroy(graph);
  rustnn_operand_destroy(output);
  rustnn_operand_destroy(scaled);
  rustnn_operand_destroy(shifted);
  rustnn_operand_destroy(scale);
  rustnn_operand_destroy(bias);
  rustnn_operand_destroy(input);
  rustnn_operand_descriptor_destroy(matrix);
  rustnn_graph_builder_destroy(builder);
  rustnn_context_destroy(context);
  return result;
}
