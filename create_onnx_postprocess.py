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
        boxes = x[:, 0:4, :].permute(0, 2, 1)
        cx, cy, w, h = boxes[..., 0:1], boxes[..., 1:2], boxes[..., 2:3], boxes[..., 3:4]
        x1 = cx - w *0.5
        y1 = cy - h *0.5
        x2 = cx + w *0.5
        y2 = cy + h *0.5
        xyxy = torch.cat([x1, y1, x2, y2], dim=2)
        yxyx = torch.cat([y1, x1, y2, x2], dim=2)
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

def create_INMSLayer(num_classes: int = 80):
    """
    ONNX NonMaxSuppression Layer oluşturma
    Bu fonksiyon, ONNX NonMaxSuppression Layer'ını oluşturur ve kaydeder.
    \n`YXYX(batch_size, num_bboxes, 4)` ve `SCORES(batch_size, num_classes, num_bboxes)` tesnorleri alır.
    """
    boxes = helper.make_tensor_value_info("boxes", onnx.TensorProto.FLOAT, [1,8400,4])
    scores = helper.make_tensor_value_info("scores", onnx.TensorProto.FLOAT, [1,num_classes,8400])

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
        inputs=["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"],
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
    input_names=["input"],
    output_names=["yxyx_out", "xyxy_out", "scores_out"],
)
onnx_model = onnx.load("onnx_folder/make_yxyx_xyxy_scores.onnx")
onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
make_yxyx_xyxy_scores_onnx, check = onnxsim.simplify(onnx_model)

create_INMSLayer()
onnx_model = onnx.load("onnx_folder/ONNX_NMS.onnx")
onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
ONNX_NMS_onnx, check = onnxsim.simplify(onnx_model)

torch.onnx.export(
    make_selected_boxes(),
    (torch.randn(100, 3).to(dtype=torch.int64), torch.randn(1, 8400, 4)),
    "onnx_folder/make_selected_boxes.onnx",
    input_names=["selected_idx", "xyxy_boxes"],
    output_names=["selected_boxes"],
    dynamic_axes={
        "selected_idx": {0: "N"},
        "selected_boxes": {0: "N"}
    }
)
onnx_model = onnx.load("onnx_folder/make_selected_boxes.onnx")
onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
make_selected_boxes_onnx, check = onnxsim.simplify(onnx_model)

# Combine the models

postprocess_onnx = combine(
    onnx_graphs=[
        make_yxyx_xyxy_scores_onnx,
        ONNX_NMS_onnx,
        make_selected_boxes_onnx,
    ],
    op_prefixes_after_merging=[
        "1",
        "2",
        "3",
    ],
    srcop_destop=[
        ["yxyx_out", "boxes", "scores_out", "scores"],
        ["2_selected_indices", "selected_idx", "1_xyxy_out", "xyxy_boxes"],
    ],
    output_onnx_file_path="onnx_folder/post_process.onnx",
)

# Check the combined model
postprocess_onnx = onnx.shape_inference.infer_shapes(postprocess_onnx)
postprocess_onnx, check = onnxsim.simplify(postprocess_onnx)
onnx.save(postprocess_onnx, "onnx_folder/post_process.onnx")
os.remove("onnx_folder/make_yxyx_xyxy_scores.onnx")
os.remove("onnx_folder/ONNX_NMS.onnx")
os.remove("onnx_folder/make_selected_boxes.onnx")
print("Postprocess ONNX model created and saved as 'onnx_folder/post_process.onnx'")

