import os
import onnx
import tensorrt
import numpy as np




engine_folder = "engine_folder"
os.makedirs(engine_folder, exist_ok=True)

# onnx_path = "onnx_folder/yolov9c.onnx"
models = [
    "onnx_folder/pre_process.onnx",
    "onnx_folder/post_process.onnx",
    "onnx_folder/yolo9_and_post_process.onnx",
    "onnx_folder/yolo11_and_post_process.onnx",
    "onnx_folder/pre_and_yolo9.onnx",
    "onnx_folder/pre_and_yolo11.onnx",
    "onnx_folder/pre_and_yolo9_and_post.onnx",
    "onnx_folder/pre_and_yolo11_and_post.onnx",
    "onnx_folder/yolo11m.onnx",
    "onnx_folder/yolov9c.onnx",
]
inputs = [
    (720,1280,3),
    (1, 84, 8400),
    (1,3,640,640),
    (1,3,640,640),
    (720,1280,3),
    (720,1280,3),
    (720,1280,3),
    (720,1280,3),
    (1,3,640,640),
    (1,3,640,640),
]
#control paths
for path in models:
    if not os.path.exists(path):
        print(f"Path {path} does not exist.")
        exit(1)

for onnx_path, input_shape in zip(models, inputs):
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