import onnxruntime
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
import os
import tensorrt
import cv2
from collections import deque

def load_engine(engine_path: str):
    """
    Load the TensorRT engine from the specified path.
    """
    with open(engine_path, "rb") as f:
        engine = f.read()
    runtime = tensorrt.Runtime(tensorrt.Logger(tensorrt.Logger.INFO))
    if engine is None:
        raise RuntimeError(f"Failed to load engine from {engine_path}")
    print(f"Successfully loaded engine from {engine_path}")
    return runtime.deserialize_cuda_engine(engine)

def allocate_buffers(engine=None):
    assert engine is not None, "engine must be provided"

    input_tensor = engine.get_tensor_name(0)
    input_shape = engine.get_tensor_shape(input_tensor)
    input_dtype = tensorrt.nptype(engine.get_tensor_dtype(input_tensor))
    # input_like = np.random.randn(*input_shape).astype(input_dtype)
    
    output_tensor = engine.get_tensor_name(1)
    output_shape = engine.get_tensor_shape(output_tensor)
    if -1 in output_shape:
        output_shape = (100,4)
    output_dtype = tensorrt.nptype(engine.get_tensor_dtype(output_tensor))
    # output_like = np.random.randn(*output_shape).astype(output_dtype)

    info=f"""
    input_tensor: {input_tensor}
    input_shape: {input_shape}
    input_dtype: {input_dtype}
    output_tensor: {output_tensor}
    output_shape: {output_shape}
    output_dtype: {output_dtype}
    """
    print(info)
    input_like = np.random.randn(*input_shape).astype(input_dtype)
    output_like = np.random.randn(*output_shape).astype(output_dtype)

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
    cuda.memset_d32_async(int(output_buffer[1]), 0, output_buffer[0].size,  stream)
    stream.synchronize()   
    return output_buffer[0]


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FPS, 60)

postprocess_session = onnxruntime.InferenceSession("onnx_folder/post_process.onnx", providers=["CPUExecutionProvider"])
pre_and_yolo_engine = load_engine("engine_folder/pre_and_yolo.engine")
in_buff, out_buff, context, stream = allocate_buffers(pre_and_yolo_engine)
name = postprocess_session.get_inputs()[0].name

lat_buf = deque(maxlen=50)
lat = 0
fps = 0
while True:

    ret, frame = cap.read()
    start1 = time.time()
    yolo_output = do_inference(context, stream, in_buff, out_buff, frame)
    end1 = time.time()
    post_output = postprocess_session.run(None, {name: yolo_output})[0]

    start2 = time.time()
    for box in post_output:
        x1, y1,x2,y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    
    info = f"latency: {lat*1000:.2f} ms, fps: {fps:.2f}"
    cv2.putText(frame, info, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    end2 = time.time()

    lat_buf.append((end1-start1) + (end2-start2))
    lat = sum(lat_buf)/len(lat_buf)
    fps = 1/lat
cap.release()
cv2.destroyAllWindows()