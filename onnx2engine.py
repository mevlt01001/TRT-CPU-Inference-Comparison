import os
import onnx
import tensorrt
import numpy as np




engine_folder = "engine_folder"
os.makedirs(engine_folder, exist_ok=True)

# onnx_path = "onnx_folder/yolov9c.onnx"

for onnx_path in ["onnx_folder/post_process.onnx"]:
    LOGGER = tensorrt.Logger(tensorrt.Logger.INFO)
    BUILDER = tensorrt.Builder(LOGGER)
    CONFIG = BUILDER.create_builder_config()
    CONFIG.set_memory_pool_limit(tensorrt.MemoryPoolType.WORKSPACE, 16*1024*1024*1024) #16gb
    NETWORK = BUILDER.create_network(1 << int(tensorrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    PARSER = tensorrt.OnnxParser(NETWORK, LOGGER)

    onnx_file = os.path.basename(onnx_path)
    file_name = onnx_file.rsplit(".")[0]
    print(f"Onnx path: {onnx_path}")
    print(f"Onnx file: {onnx_file}")
    print(f"File name: {file_name}")
    input_shape = (720, 1280, 3) if (file_name == "pre_and_yolo" or file_name == "pre_and_yolo_and_post") else (1,84,8400)
    print(f"Input shape: {input_shape}")

    with open(onnx_path, "rb") as f:
        if not PARSER.parse(f.read()):
            print("ERROR: Failed to parse ONNX file:", onnx_file)
            for error in range(PARSER.num_errors):
                print("ERROR:", PARSER.get_error(error))
            exit(1)

    print("Successfully parsed ONNX file:", onnx_file)

    if BUILDER.platform_has_fast_fp16:
        CONFIG.set_flag(tensorrt.BuilderFlag.FP16)

    if input_shape is not None:
        PROFILE = BUILDER.create_optimization_profile()
        PROFILE.set_shape(NETWORK.get_input(0).name, input_shape, input_shape, input_shape) 
        CONFIG.add_optimization_profile(PROFILE)

    engine = BUILDER.build_serialized_network(NETWORK, CONFIG)
    if engine is None:
        print("ERROR: Failed to build engine for ONNX file:", onnx_file)
        exit(1)

    print("Successfully built engine for ONNX file:", onnx_file)
    engine_path = os.path.join(engine_folder, file_name + ".engine")
    with open(engine_path, "wb") as f:
        f.write(engine)
    print("Successfully saved engine file:", engine_path)