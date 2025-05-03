import tensorrt
import onnxruntime
import numpy as np
import time
import os
import pycuda.driver as cuda
import pycuda.autoinit

def load_engine(engine_path: str):
    """
    Load the TensorRT engine from the specified path.
    """
    with open(engine_path, "rb") as f:
        engine = f.read()
    runtime = tensorrt.Runtime(tensorrt.Logger(tensorrt.Logger.INFO))
    return runtime.deserialize_cuda_engine(engine)

def allocate_buffers(engine=None, input_like: np.ndarray=None, output_like: np.ndarray=None):
    assert engine is not None, "engine must be provided"
    assert input_like is not None, "input_shape must be provided"
    assert output_like is not None, "output_shape must be provided"

    CPU_INPUT_BUFFER = cuda.pagelocked_empty_like(input_like)
    CPU_OUTPUT_BUFFER = cuda.pagelocked_empty_like(output_like)
    GPU_INPUT_BUFFER = cuda.mem_alloc_like(CPU_INPUT_BUFFER)
    GPU_OUTPUT_BUFFER = cuda.mem_alloc_like(CPU_OUTPUT_BUFFER)

    context = engine.create_execution_context()
    context.set_tensor_address(engine.get_tensor_name(0), int(GPU_INPUT_BUFFER))
    context.set_tensor_address(engine.get_tensor_name(1), int(GPU_OUTPUT_BUFFER))
    stream = cuda.Stream()
    
    return [CPU_INPUT_BUFFER, GPU_INPUT_BUFFER], [CPU_OUTPUT_BUFFER, GPU_OUTPUT_BUFFER], context, stream

def do_inference(context, stream, input_buffer, output_buffer, data):
    """
    Run inference on the input buffer and return the output buffer.
    """
    input_buffer[0][:] = data
    cuda.memcpy_htod_async(input_buffer[1], input_buffer[0], stream)
    context.execute_async_v3(stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(output_buffer[0], output_buffer[1], stream)
    return output_buffer[0]

def measure_latency(pre_process_type: str=None, post_process_type: str=None):
    """
    default onnx path: onnx_folder/\n
    default engine path: engine_folder/
    """
    assert pre_process_type in ["gpu", "cpu", "trt"], f"pre_preocess_type must be in ['gpu', 'cpu', 'trt'], but got {pre_process_type}"
    assert post_process_type in ["gpu", "cpu", "trt"], f"post_process_type must be in ['gpu', 'cpu', 'trt'], but got {post_process_type}"

    pre_process_type = {
        "gpu": "CUDAExecutionProvider",
        "cpu": "CPUExecutionProvider",
    }

    post_process_type = {
        "gpu": "CUDAExecutionProvider",
        "cpu": "CPUExecutionProvider",
    }

    yolov9c_engine_path = "engine_folder/yolov9c.engine"
    yolov9c_and_postprocess_engine_path = "engine_folder/yolo_and_post_process.engine"
    pre_preocess_and_yolov9c_engine_path = "engine_folder/pre_and_yolo.engine"
    pre_preocess_and_yolov9c_and_postprocess_engine_path = "engine_folder/pre_and_yolo_and_post_process.engine"

    preprocess_onnx_path = "onnx_folder/preprocess.onnx"
    postprocess_onnx_path = "onnx_folder/post_process.onnx"


    if pre_process_type != "trt" and post_process_type != "trt":
        # Sadece YOLOv9c engine formatında diğerleri onnxruntime ile çalışacak
        pre_preocess_session = onnxruntime.InferenceSession(preprocess_onnx_path, providers=[pre_process_type])
        post_process_session = onnxruntime.InferenceSession(postprocess_onnx_path, providers=[post_process_type])
        yolov9_engine = load_engine(yolov9c_engine_path)
        input_like = np.random.randn(1, 3, 640, 640).astype(np.float32)
        output_like = np.random.randn(1, 8400, 85).astype(np.float32)
        input_buff, output_buff, context, stream = allocate_buffers(yolov9_engine, input_like, output_like)

        # Measure latency
        
        



    dummy_input = np.random.randn(720, 1280, 3).astype(np.float32)

    