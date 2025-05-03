import onnx
import torch
import onnxsim
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper
import numpy as np
import os
from snc4onnx import combine

os.makedirs("onnx_folder", exist_ok=True)

class make_yxyx_xyxy_scores(torch.nn.Module):
    def __init__(self, num_classes: int = 80):
        super(make_yxyx_xyxy_scores, self).__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor):
        # x.shape is (1,84,8400)
        scores = x[:, 4:self.num_classes+4, :]
        cx, cy, w, h = x[:, 0:4, :].split(1, dim=1)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        xyxy = torch.cat([x1, y1, x2, y2], dim=1).permute(0, 2, 1)
        yxyx = torch.cat([y1, x1, y2, x2], dim=1).permute(0, 2, 1)
        return yxyx, xyxy, scores

class make_selected_boxes(torch.nn.Module):
    def __init__(self):
        super(make_selected_boxes, self).__init__()

    def forward(self, selected_indices: torch.Tensor, xyxy_boxes: torch.Tensor):
        # selected_indices.shape is (3, N)
        # xyxy_boxes.shape is (1, 8400, 4)
        # selected_indices[0] is batch index
        # selected_indices[1] is class index
        # selected_indices[2] is box index

        selected_indices = selected_indices[:,2]
        selected_boxes = xyxy_boxes[0, selected_indices, :]
        return selected_boxes

def create_INMSLayer():
    """
    ONNX NonMaxSuppression Layer oluşturma
    Bu fonksiyon, ONNX NonMaxSuppression Layer'ını oluşturur ve kaydeder.
    \n`YXYX(batch_size, num_bboxes, 4)` ve `SCORES(batch_size, num_classes, num_bboxes)` tesnorleri alır.
    """
    boxes = helper.make_tensor_value_info("Boxes", onnx.TensorProto.FLOAT, [1,8400,4])
    scores = helper.make_tensor_value_info("Scores", onnx.TensorProto.FLOAT, [1,1,8400])

    max_output_boxes_per_class = helper.make_tensor(
        name="max_output_boxes_per_class",
        data_type=onnx.TensorProto.INT64,
        dims=[1],
        vals=[100]
    )

    iou_threshold = helper.make_tensor(
        name="iou_threshold",
        data_type=onnx.TensorProto.FLOAT,
        dims=[1],
        vals=[0.55]
    )

    score_threshold = helper.make_tensor(
        name="score_threshold",
        data_type=onnx.TensorProto.FLOAT,
        dims=[1],
        vals=[0.25]
    )

    nms_node = helper.make_node(
        "NonMaxSuppression",
        inputs=["Boxes", "Scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"],
        outputs=["selected_indices"],
        name="NMS",
        center_point_box=0
    )

    output = helper.make_tensor_value_info("selected_indices", onnx.TensorProto.INT64, ['N', 3])

    graph = helper.make_graph(
        nodes=[nms_node],
        name="nms_graph",
        inputs=[boxes, scores],
        outputs=[output],
        initializer=[iou_threshold, score_threshold, max_output_boxes_per_class]
    )

    # Model oluştur
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
    model.ir_version = 9

    model, check = onnxsim.simplify(model)
    onnx.save(model, "onnx_folder/ONNX_NMS.onnx")

torch.onnx.export(
    make_yxyx_xyxy_scores(),
    torch.randn(1, 84, 8400),
    "onnx_folder/make_yxyx_xyxy_scores.onnx",
    input_names=["make_yxyx_xyxy_scores_input"],
    output_names=["yxyx", "xyxy", "scores"],
)
onnx_model = onnx.load("onnx_folder/make_yxyx_xyxy_scores.onnx")
onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
make_yxyx_xyxy_scores_onnx, check = onnxsim.simplify(onnx_model)
onnx.save(onnx_model, "onnx_folder/make_yxyx_xyxy_scores.onnx")

create_INMSLayer()
onnx_model = onnx.load("onnx_folder/ONNX_NMS.onnx")
onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
ONNX_NMS_onnx, check = onnxsim.simplify(onnx_model)
onnx.save(onnx_model, "onnx_folder/ONNX_NMS.onnx")

torch.onnx.export(
    make_selected_boxes(),
    (torch.randn(100, 3).to(dtype=torch.int64), torch.randn(1, 8400, 4)),
    "onnx_folder/make_selected_boxes.onnx",
    input_names=["_selected_indices", "xyxy_boxes"],
    output_names=["selected_boxes"],
    dynamic_axes={
        "_selected_indices": {0: "N"},
        "xyxy_boxes": {0: "N"}
    }
)
onnx_model = onnx.load("onnx_folder/make_selected_boxes.onnx")
onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
make_selected_boxes_onnx, check = onnxsim.simplify(onnx_model)
onnx.save(onnx_model, "onnx_folder/make_selected_boxes.onnx")

# Combine the models

postprocess_onnx = combine(
    onnx_graphs=[
        make_yxyx_xyxy_scores_onnx,
        ONNX_NMS_onnx,
        make_selected_boxes_onnx,
    ],
    srcop_destop=[
        ["yxyx", "Boxes", "scores", "Scores"],
        ["selected_indices", "_selected_indices", "xyxy", "xyxy_boxes"],
    ],
    output_onnx_file_path="onnx_folder/postprocess.onnx",
)

# Check the combined model
postprocess_onnx = onnx.shape_inference.infer_shapes(postprocess_onnx)
postprocess_onnx, check = onnxsim.simplify(postprocess_onnx)
onnx.save(postprocess_onnx, "onnx_folder/postprocess.onnx")
os.remove("onnx_folder/make_yxyx_xyxy_scores.onnx")
os.remove("onnx_folder/ONNX_NMS.onnx")
os.remove("onnx_folder/make_selected_boxes.onnx")
print("Postprocess ONNX model created and saved as 'onnx_folder/postprocess.onnx'")

