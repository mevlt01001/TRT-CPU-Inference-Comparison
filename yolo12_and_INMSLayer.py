from utils.trt import *
import pycuda.autoinit
import numpy as np
import time
import cv2

yolo = load_engine("EngineFolder/YOLO12_for_inms_layer.engine")
inms_layer = load_engine("EngineFolder/INMSLayer_trt.engine")
yolo_input_buffers, yolo_output_buffers = allocate_buffers(yolo)
inms_layer_input_buffers, inms_layer_output_buffers = allocate_buffers(inms_layer, outshape=(100, 3))
yolo_context, yolo_stream = create_execution_context(yolo, yolo_input_buffers, yolo_output_buffers)
inms_layer_context, inms_layer_stream = create_execution_context(inms_layer, inms_layer_input_buffers, inms_layer_output_buffers)
latencies = []

for i in range(150):
    print(i)
    img = cv2.imread("273271-2b427000e2a2b025_jpg.rf.7d933851f233dcd09cf166e310a4b407.jpg") # fotoğrafta 15 kişi var
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    data = np.transpose(img, (2, 0, 1)).astype(np.float32)
    data_for_yolo = np.expand_dims(data, axis=0)/255.0
    #data_for_yolo = np.random.rand(1, 3, 640, 640).astype(np.float32)
    start = time.time()
    out = run_inference(yolo_context, yolo_stream, yolo_input_buffers, yolo_output_buffers, data=data_for_yolo)
    max_output_boxes = np.array([100], dtype=np.int32)
    iou_threshold = np.array([0.55], dtype=np.float32)
    score_threshold = np.array([0.9], dtype=np.float32)
    data_for_inms = [np.array(yolo_output_buffers[0].cpu_buffer).reshape(1, 1000, 4), np.array(yolo_output_buffers[1].cpu_buffer).reshape(1, 1000, 1), max_output_boxes, iou_threshold, score_threshold]
    out=run_inference(inms_layer_context, inms_layer_stream, inms_layer_input_buffers, inms_layer_output_buffers, data=data_for_inms)
    print(np.trim_zeros(out[0], "b").reshape(-1, 3).shape)#53 bbox veriyor yanı 53-15 tane fp box var.
    latencies.append(time.time() - start)
total_time = sum(latencies)

print(f"TRT FPS: {len(latencies) / total_time:.2f}")

"""
DESKTOP TRT FPS: 47.35
AGX ORIN TRT FPS: None
"""