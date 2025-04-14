import onnx
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper
import numpy as np

# Giriş tensorleri
boxes = helper.make_tensor_value_info("Boxes", onnx.TensorProto.FLOAT, [1, 8400, 4])
scores = helper.make_tensor_value_info("Scores", onnx.TensorProto.FLOAT, [1, 8400, 1])

# Sabit parametre tensorleri
max_output_boxes_per_class = helper.make_tensor(
    name="max_output_boxes_per_class",
    data_type=onnx.TensorProto.INT64,
    dims=[],
    vals=[100]
)

iou_threshold = helper.make_tensor(
    name="iou_threshold",
    data_type=onnx.TensorProto.FLOAT,
    dims=[],
    vals=[0.5]
)

score_threshold = helper.make_tensor(
    name="score_threshold",
    data_type=onnx.TensorProto.FLOAT,
    dims=[],
    vals=[0.25]
)

# NMS düğümü
nms_node = helper.make_node(
    "NonMaxSuppression",
    inputs=["Boxes", "Scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"],
    outputs=["selected_indices"],
    name="NMS",
    center_point_box=1
)

# Çıktı
output = helper.make_tensor_value_info("selected_indices", onnx.TensorProto.INT64, [None, 3])

# Graph oluştur
graph = helper.make_graph(
    nodes=[nms_node],
    name="nms_graph",
    inputs=[boxes, scores],
    outputs=[output],
    initializer=[max_output_boxes_per_class, iou_threshold, score_threshold]
)

# Model oluştur
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
onnx.save(model, "utils/nms_only.onnx")
print(model)
