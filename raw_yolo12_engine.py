from utils.trt import *
import pycuda.autoinit
import numpy as np
import time

engine = load_engine("EngineFolder/yolo12l.engine")
input_buffers, output_buffers = allocate_buffers(engine)
context, stream = create_execution_context(engine, input_buffers, output_buffers)
latencies = []
for i in range(100):
    print(i)
    data = np.random.rand(1, 3, 640, 640).astype(np.float32)
    start = time.time()
    run_inference(context, stream, input_buffers, output_buffers, data)
    latencies.append(time.time() - start)
print(f"TRT FPS: {len(latencies) / sum(latencies):.2f}")

"""
TRT FPS: 49.04
"""