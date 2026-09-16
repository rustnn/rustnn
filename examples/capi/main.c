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
  RustnnGraphBuilder *builder = NULL;
  RustnnOperandDescriptor *matrix = NULL;
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

  if (!check(rustnn_graph_builder_create_uncompiled(&builder)) ||
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
  result = EXIT_SUCCESS;

cleanup:
  rustnn_string_destroy(webnn_text);
  rustnn_operand_destroy(output);
  rustnn_operand_destroy(scaled);
  rustnn_operand_destroy(shifted);
  rustnn_operand_destroy(scale);
  rustnn_operand_destroy(bias);
  rustnn_operand_destroy(input);
  rustnn_operand_descriptor_destroy(matrix);
  rustnn_graph_builder_destroy(builder);
  return result;
}
