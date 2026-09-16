#include <rustnn/rustnn.hpp>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

int main() {
  try {
    rustnn::initializeLogger();

    rustnn::Context context;
    rustnn::GraphBuilder builder(context);
    rustnn::OperandDescriptor matrix(rustnn::DataType::Float32, {2, 2});

    auto input = builder.input("input", matrix);
    auto bias = builder.constant<float>(matrix, {1.0f, 2.0f, 3.0f, 4.0f});
    auto scale = builder.constant<float>(matrix, {2.0f, 2.0f, 2.0f, 2.0f});

    auto shifted = builder.add(input, bias, rustnn::OperatorOptions{"add bias"});
    auto scaled =
        builder.multiply(shifted, scale, rustnn::OperatorOptions{"scale values"});
    auto output = builder.relu(scaled, rustnn::OperatorOptions{"clamp negatives"});

    const auto shape = builder.shape(output);
    std::cout << "output shape:";
    for (std::uint64_t dimension : shape) {
      std::cout << ' ' << dimension;
    }
    std::cout << "\n\n" << builder.webnnText({{"output", &output}}) << '\n';

    auto graph = builder.build({{"output", &output}});
    rustnn::TensorDescriptor input_descriptor(rustnn::DataType::Float32, {2, 2},
                                              false, true);
    rustnn::TensorDescriptor output_descriptor(rustnn::DataType::Float32, {2, 2},
                                               true, false);
    auto input_tensor = context.createTensor(input_descriptor);
    auto output_tensor = context.createTensor(output_descriptor);

    context.writeTensor<float>(input_tensor, {1.0f, -3.0f, 2.0f, -10.0f});
    context.dispatch(graph, {{"input", &input_tensor}}, {{"output", &output_tensor}});
    const auto values = context.readTensor<float>(output_tensor, 4);
    const std::vector<float> expected = {4.0f, 0.0f, 10.0f, 0.0f};
    if (values != expected) {
      std::cerr << "unexpected output values\n";
      return EXIT_FAILURE;
    }

    std::cout << "dispatch output:";
    for (float value : values) {
      std::cout << ' ' << value;
    }
    std::cout << '\n';
    return 0;
  } catch (const rustnn::Error &error) {
    std::cerr << "rustnn error: " << error.what() << '\n';
    return 1;
  }
}
