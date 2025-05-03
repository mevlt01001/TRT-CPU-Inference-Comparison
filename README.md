ONNX formatına çevrilmiş [preprocess](create_onnx_preprocess.py) ve [postprocess](create_onnx_postprocess.py) katmanlarını; **Jetson AGX Orin** için ONNXRuntime(CPU,GPU) ve TensorRT FP16 optimizasyonlu formatlarda **YOLOv9c** ile birlikte gecikmelerini hesaplanır.

*PINTO0309 WhloBody28-refine çalışmasından dolayı **YOLOv9c** esas alındı.*

*[load YOLOv9c.onnx](https://drive.google.com/drive/folders/1Y4fIZ2RIcwwvMGylCMhVJTgsWY8OH8LE?usp=drive_link)*

[create_onnx_preprocess.py](create_onnx_preprocess.py) ONNX formatında preprocess layer oluşturur.\
[create_onnx_postprocess.py](create_onnx_postprocess.py) ONNX formatında postprocess layer oluşturur.\
[make_yolo_pre_post_onnx.py](make_yolo_pre_post_onnx.py) ONNX formatında preprocess+YOLO, YOLO+postprocess ve preprocess+YOLO+postprocess şeklinde 3 model oluşturur.


## ONNXRuntime ölçüm sonuçları:
(80 COCO class)
- preprocess on gpu: 5.95ms
- preprocess on cpu: 24.4ms
- postprocess on gpu: 123 ms
- postprocess on cpu: 23.5 ms

## cv2.dnn.NMSBoxes sonucu:
- bboxes = np.random.rand(8400, 4).astype(np.float32).tolist()
- scores = np.random.rand(8400).astype(np.float32).tolist()
- cv2.dnn.NMSBoxes(bboxes, scores, 0.5, 0.4)
- **14.7ms**


