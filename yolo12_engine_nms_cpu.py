from utils.trt import *
from utils.NMS import YOLO12_postprocess
import pycuda.autoinit
import numpy as np
import torch
import time

engine = load_engine("EngineFolder/yolo12l.engine")
input_buffers, output_buffers = allocate_buffers(engine)
context, stream = create_execution_context(engine, input_buffers, output_buffers)
trt_latencies = []
for i in range(100):
    print(i)
    data = np.random.rand(1, 3, 640, 640).astype(np.float32)
    start = time.time()
    run_inference(context, stream, input_buffers, output_buffers, data)
    trt_latencies.append(time.time() - start)
total_trt_time = sum(trt_latencies)

cpu_latencies = []
for i in range(100):
    print(i)
    data = torch.randn(1, 84, 8400)
    start = time.time()
    YOLO12_postprocess(score_threshold=0.25, iou_threshold=0.45).forward(data)
    cpu_latencies.append(time.time() - start)
total_cpu_time = sum(cpu_latencies)

print(f"TRT FPS: {len(trt_latencies) / total_trt_time:.2f}")
print(f"CPU FPS: {len(cpu_latencies) / total_cpu_time:.2f}")
print(f"TRT+CPU FPS: {(len(trt_latencies) + len(cpu_latencies)) / (total_trt_time + total_cpu_time):.2f}")

"""
DESKTOP TRT FPS: 48.27 (Raw_model)
DESKTOP CPU FPS: 39.85 (NSM)
DESKTOP TRT+CPU FPS: 43.66

AGX ORIN TRT FPS: 37.43 (Raw_model))
AGX ORIN CPU FPS: 29.84 (NMS)
AGX ORIN TRT+CPU FPS: 33.21
"""