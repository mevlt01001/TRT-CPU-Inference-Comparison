from utils.trt import *
import pycuda.autoinit
import numpy as np
import time

engine = load_engine("EngineFolder/ONNX_NMS.engine")
input_buffers, output_buffers = allocate_buffers(engine, outshape=(100, 5))
context, stream = create_execution_context(engine, input_buffers, output_buffers)
latencies = []
for i in range(1000):
    print(i)
    boxes = np.random.rand(1,8400,4).astype(np.float32)
    scores = np.random.rand(1,1,8400).astype(np.float32)
    data = [boxes, scores]
    start = time.time()
    run_inference(context, stream, input_buffers, output_buffers, data)
    latencies.append(time.time() - start)
total_time = sum(latencies)

print(f"TRT FPS: {len(latencies) / total_time:.2f}")

"""
DESKTOP TRT FPS: 1382.06
AGX ORIN TRT FPS: None
"""