import torch
import onnx
import onnxsim
import os

class PreProcess(torch.nn.Module):
    def __init__(self):
        super(PreProcess, self).__init__()

    def forward(self, x: torch.Tensor):
        #x.shape is (720, 1280, 3)
        # (720, 1280, 3) -> (3, 720, 1280)
        x = x.permute(2, 0, 1)
        B,G,R = x[0:1, :, :], x[1:2, :, :], x[2:3, :, :]
        # BGR -> RGB
        x = torch.cat((R, G, B), dim=0)
        # (3, 720, 1280) -> (1, 3, 720, 1280)
        x = x.unsqueeze(0)
        # (1, 3, 720, 1280) -> (1, 3, 640, 640)
        x = torch.nn.functional.interpolate(
            input=x,
            size=(640, 640),
        )
        x = x * 0.003921569 # / 255.0
        return x

os.makedirs("onnx_folder", exist_ok=True)

torch.onnx.export(
    PreProcess(),
    torch.randn(720, 1280,3),
    "onnx_folder/pre_process.onnx",
    input_names=["pre_OP_input"],
    output_names=["pre_OP_output"],
    dynamic_axes={
        "pre_OP_input": {0: "H", 1: "W"},
    },
)

pre_onnx = onnx.load("onnx_folder/pre_process.onnx")
pre_onnx, check = onnxsim.simplify(pre_onnx)
pre_onnx = onnx.shape_inference.infer_shapes(pre_onnx)
onnx.save(pre_onnx, "onnx_folder/pre_process.onnx")