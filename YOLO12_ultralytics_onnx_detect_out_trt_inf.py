from utils.trt import *
import pycuda.autoinit
import numpy as np
import time, cv2, matplotlib.pyplot as plt

engine = load_engine("EngineFolder/yolo12l_detect_yolo.engine")
input_buffers, output_buffers = allocate_buffers(engine, outshape=(100, 5))
context, stream = create_execution_context(engine, input_buffers, output_buffers)
latencies = []
out = []
for i in range(130):
    print(i)
    img = cv2.imread("273271-2b427000e2a2b025_jpg.rf.7d933851f233dcd09cf166e310a4b407.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    data = np.transpose(img, (2, 0, 1)).astype(np.float32)
    data = np.expand_dims(data, axis=0)/255.0
    #data = np.random.rand(1, 3, 640, 640).astype(np.float32)
    start = time.time()
    _out=run_inference(context, stream, input_buffers, output_buffers, data)[0]
    latencies.append(time.time() - start)
    out.append(np.trim_zeros(_out, "b").reshape(-1, 6).tolist())
total_time = sum(latencies)

out = out[-1]


for box in out:
    cx, cy, w, h, _, _ = box
    # x1, y1 = int((cx - w / 2)), int((cy - h / 2))
    # x2, y2 = int((cx + w / 2)), int((cy + h / 2))
    x1, y1, x2, y2 = int(cx), int(cy), int(w), int(h)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

plt.imshow(img)
plt.show()

print(f"TRT FPS: {len(latencies) / total_time:.2f}")

"""
DESKTOP TRT FPS: 32.84
AGX ORIN TRT FPS: None
"""