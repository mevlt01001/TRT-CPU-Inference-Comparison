from utils.trt import *
import pycuda.autoinit
import numpy as np
import time

engine = load_engine("EngineFolder/YOLO12_INMSLayer.engine")
input_buffers, output_buffers = allocate_buffers(engine, outshape=(100, 5))
context, stream = create_execution_context(engine, input_buffers, output_buffers)
latencies = []
for i in range(100):
    print(i)
    data = np.random.rand(1, 3, 640, 640).astype(np.float32)
    start = time.time()
    run_inference(context, stream, input_buffers, output_buffers, data)
    latencies.append(time.time() - start)
total_time = sum(latencies)

print(f"TRT FPS: {len(latencies) / total_time:.2f}")

"""
DESKTOP TRT FPS: 33.57
AGX ORIN TRT FPS: None
"""