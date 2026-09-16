#include <rustnn/rustnn.hpp>

#include <cstdint>
#include <iostream>
#include <vector>

int main() {
  try {
    rustnn::GraphBuilder builder;
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
    return 0;
  } catch (const rustnn::Error &error) {
    std::cerr << "rustnn error: " << error.what() << '\n';
    return 1;
  }
}
