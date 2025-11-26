CARGO := cargo
DOT ?= dot
GRAPH_FILE ?= examples/sample_graph.json
DOT_PATH ?= target/graph.dot
PNG_PATH ?= target/graph.png
ONNX_PATH ?= target/graph.onnx
COREML_PATH ?= target/graph.mlmodel
COREMLC_PATH ?= target/graph.mlmodelc
ORT_VERSION ?= 1.17.0
ORT_BASE ?= https://github.com/microsoft/onnxruntime/releases/download/v$(ORT_VERSION)
ORT_TARBALL ?= onnxruntime-osx-arm64-$(ORT_VERSION).tgz
ORT_DIR ?= target/onnxruntime
ORT_LIB_DIR ?= $(ORT_DIR)/onnxruntime-osx-arm64-$(ORT_VERSION)/lib
ORT_LIB_LOCATION ?= $(ORT_LIB_DIR)
.PHONY: build test fmt run viz onnx coreml coreml-validate onnx-validate validate-all-env

clean:
	$(CARGO) clean
	rm -f target/graph.dot target/graph.png target/graph.onnx target/graph.mlmodel
	rm -rf target/graph.mlmodelc
	rm -rf $(ORT_DIR)

build:
	$(CARGO) build

test:
	$(CARGO) test

fmt:
	$(CARGO) fmt

run:
	$(CARGO) run -- examples/sample_graph.json

viz:
	$(CARGO) run -- $(GRAPH_FILE) --export-dot $(DOT_PATH)
	$(DOT) -Tpng $(DOT_PATH) -o $(PNG_PATH)
	open $(PNG_PATH)


onnxruntime-download:
	mkdir -p $(ORT_DIR)
	curl -L $(ORT_BASE)/$(ORT_TARBALL) -o $(ORT_DIR)/$(ORT_TARBALL)
	tar -xzf $(ORT_DIR)/$(ORT_TARBALL) -C $(ORT_DIR)

onnx: onnxruntime-download
	ORT_STRATEGY=system ORT_LIB_LOCATION=$(ORT_LIB_LOCATION) DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) RUSTFLAGS="-L $(ORT_LIB_DIR)" $(CARGO) run --features onnx-runtime -- $(GRAPH_FILE) --convert onnx --convert-output $(ONNX_PATH)
	@echo "ONNX graph written to $(ONNX_PATH)"

onnx-validate: onnx
	ORT_STRATEGY=system ORT_LIB_LOCATION=$(ORT_LIB_LOCATION) DYLD_LIBRARY_PATH=$(ORT_LIB_DIR) RUSTFLAGS="-L $(ORT_LIB_DIR)" $(CARGO) run --features onnx-runtime -- $(GRAPH_FILE) --convert onnx --convert-output $(ONNX_PATH) --run-onnx

coreml:
	$(CARGO) run -- $(GRAPH_FILE) --convert coreml --convert-output $(COREML_PATH)
	@echo "CoreML graph written to $(COREML_PATH)"

coreml-validate: coreml
	$(CARGO) run --features coreml-runtime -- $(GRAPH_FILE) --convert coreml --convert-output $(COREML_PATH) --run-coreml --coreml-compiled-output $(COREMLC_PATH)

webnn-venv:
	python3 -m venv .venv-webnn
	. .venv-webnn/bin/activate && pip install --upgrade pip && pip install git+https://github.com/huningxin/onnx2webnn.git

validate-all-env: build test onnx-validate coreml-validate
	@echo "Full pipeline (build/test/convert/validate) completed."
