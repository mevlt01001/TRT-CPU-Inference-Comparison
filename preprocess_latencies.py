import onnxruntime
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
import os
import tensorrt


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
def measure_latency(onnx_path:str=None, type: str=None, data: np.ndarray=None):
    """
    default onnx path: onnx_folder/\n
    default engine path: engine_folder/
    """
    assert data is not None, "data must be provided"
    assert onnx_path is not None, "onnx_path must be provided"
    assert type is not None, "pre_process_type must be provided"
    assert os.path.exists(onnx_path), f"onnx_path {onnx_path} does not exist"
    assert type in ["gpu", "cpu", "trt"], f"pre_preocess_type must be in ['gpu', 'cpu', 'trt'], but got {type}"

    providers = {
        "gpu": ["CUDAExecutionProvider"],
        "cpu": ["CPUExecutionProvider"],
    }
    session = onnxruntime.InferenceSession(onnx_path, providers=providers[type])
    input_name = session.get_inputs()[0].name

    cnt=0
    start = time.time()
    while True:
        cnt+=1
        output = session.run(None, {input_name: data})
        print(f"{onnx_path} latency in {type} mode {(time.time()-start)/cnt*1000:.2f} ms")
    
if __name__ == "__main__":
    # measure_latency(onnx_path="onnx_folder/pre_process.onnx",type="gpu", data=np.random.rand(720, 1280,3).astype(np.float32))
    import cv2
    bboxes = np.random.rand(8400, 4).astype(np.float32).tolist()
    scores = np.random.rand(8400).astype(np.float32).tolist()
    cnt = 0
    start = time.time()
    while True:
        cnt+=1
        selected_indices = cv2.dnn.NMSBoxes(bboxes, scores, 0.5, 0.4)
        print(selected_indices)
        print(f"latency in cv2.dnn.nms mode {(time.time()-start)/cnt*1000:.2f} ms")
        