from utils.trt import *
import pycuda.autoinit
import numpy as np
import time

engine = load_engine("EngineFolder/INMSLayer.engine")
input_buffers, output_buffers = allocate_buffers(engine, outshape=(10, 3))
context, stream = create_execution_context(engine, input_buffers, output_buffers)
latencies = []
for i in range(1000):
    print(i)
    boxes = np.random.rand(1, 8400, 4).astype(np.float32)
    scores = np.random.rand(1, 8400, 1).astype(np.float32)
    max_output_boxes = np.array([100], dtype=np.int32)
    iou_threshold = np.array([0.45], dtype=np.float32)
    score_threshold = np.array([0.25], dtype=np.float32)
    data = [boxes, scores, max_output_boxes, iou_threshold, score_threshold]
    start = time.time()
    run_inference(context, stream, input_buffers, output_buffers, data=data)
    latencies.append(time.time() - start)
total_time = sum(latencies)

print(f"TRT FPS: {len(latencies) / total_time:.2f}")

"""
DESKTOP TRT FPS: 1293.98
AGX ORIN TRT FPS: None
"""