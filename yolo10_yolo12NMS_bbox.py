import matplotlib.pyplot as plt
import numpy as np
import cv2
from utils.trt import *
import pycuda.autoinit

image_path = "crowd_human.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (640, 640))
data = np.transpose(image, (2, 0, 1))
data = np.expand_dims(data, axis=0)
data = data.astype(np.float32)/255.0

engine_yolo10 = load_engine("EngineFolder/yolov10l.engine")
yolo10_input_buffers, yolo10_output_buffers = allocate_buffers(engine_yolo10)
context_yolo10, stream_yolo10 = create_execution_context(engine_yolo10, yolo10_input_buffers, yolo10_output_buffers)

engine_yolo12 = load_engine("EngineFolder/Y_0.22_0.55.engine")
yolo12_input_buffers, yolo12_output_buffers = allocate_buffers(engine_yolo12, outshape=(100, 5))
context_yolo12, stream_yolo12 = create_execution_context(engine_yolo12, yolo12_input_buffers, yolo12_output_buffers)

yolo10_out = run_inference(context_yolo10, stream_yolo10, yolo10_input_buffers, yolo10_output_buffers, data)[0]
yolo12_out = run_inference(context_yolo12, stream_yolo12, yolo12_input_buffers, yolo12_output_buffers, data)[0]

yolo12_out = np.trim_zeros(yolo12_out, trim='b')
yolo12_out = yolo12_out.reshape(-1,5)
yolo10_out = yolo10_out.reshape(300,6)
yolo10_out = yolo10_out[yolo10_out[..., 4] > 0.1]
yolo10_out = yolo10_out[yolo10_out[..., 5] == 0]


image_10 = cv2.imread(image_path)
image_10 = cv2.cvtColor(image_10, cv2.COLOR_BGR2RGB)
image_10 = cv2.resize(image, (640, 640))
for box in yolo10_out:
    x1, y1, x2, y2, conf, cls = box.astype(np.int32)
    cv2.rectangle(image_10, (x1, y1), (x2, y2), (255, 0, 0), 2)

image_12 = cv2.imread(image_path)
image_12 = cv2.cvtColor(image_12, cv2.COLOR_BGR2RGB)
image_12 = cv2.resize(image, (640, 640))
for box in yolo12_out:
    x1, y1, x2, y2, conf = box.astype(np.int32)
    cv2.rectangle(image_12, (x1, y1), (x2, y2), (0, 255, 0), 2)

plt.figure(figsize=(12, 6))
plt.axis('off')
plt.suptitle("YOLO10 vs YOLO12 Ensemble NMS", fontsize=16)
plt.title(f"YOLO10(score > 0.1) {yolo10_out.shape[0]} Bbox                                                                  YOLO12(score > 0.22 and iou > 0.55) {yolo12_out.shape[0]} Bbox", fontsize=12)
plt.subplot(1, 2, 1)
plt.imshow(image_10)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(image_12)
plt.axis('off')
plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.savefig("yolo10_yolo12NMS_bbox.png", dpi=300)
plt.show()