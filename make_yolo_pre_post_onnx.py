import onnx 
import onnxsim
import snc4onnx

yolov9c_onnx = onnx.load("onnx_folder/yolov9c.onnx")
yolov9c_onnx = onnx.shape_inference.infer_shapes(yolov9c_onnx)
yolov9c_onnx, check = onnxsim.simplify(yolov9c_onnx)

postprocess_onnx = onnx.load("onnx_folder/post_process.onnx")
postprocess_onnx = onnx.shape_inference.infer_shapes(postprocess_onnx)
postprocess_onnx, check = onnxsim.simplify(postprocess_onnx)

preprocess_onnx = onnx.load("onnx_folder/pre_process.onnx")
preprocess_onnx = onnx.shape_inference.infer_shapes(preprocess_onnx)
preprocess_onnx, check = onnxsim.simplify(preprocess_onnx)

yolo_and_post_onnx = snc4onnx.combine(
    onnx_graphs=[
        yolov9c_onnx,
        postprocess_onnx,
    ],
    op_prefixes_after_merging=[
        "1",
        "2",
    ],
    srcop_destop=[
        [yolov9c_onnx.graph.output[0].name, postprocess_onnx.graph.input[0].name],
    ],
    output_onnx_file_path="onnx_folder/yolo_and_post_process.onnx",
)
yolo_and_post_onnx = onnx.shape_inference.infer_shapes(yolo_and_post_onnx)
yolo_and_post_onnx, check = onnxsim.simplify(yolo_and_post_onnx)
onnx.save(yolo_and_post_onnx, "onnx_folder/yolo_and_post_process.onnx")
print("Successfully saved yolo_and_post_process.onnx")

pre_and_yolo = snc4onnx.combine(
    onnx_graphs=[
        preprocess_onnx,
        yolov9c_onnx,
    ],
    op_prefixes_after_merging=[
        "1",
        "2",
    ],
    srcop_destop=[
        [preprocess_onnx.graph.output[0].name, yolov9c_onnx.graph.input[0].name],
    ],
    output_onnx_file_path="onnx_folder/pre_and_yolo.onnx",
)
pre_and_yolo = onnx.shape_inference.infer_shapes(pre_and_yolo)
pre_and_yolo, check = onnxsim.simplify(pre_and_yolo)
onnx.save(pre_and_yolo, "onnx_folder/pre_and_yolo.onnx")
print("Successfully saved pre_and_yolo.onnx")

pre_and_yolo_and_post = snc4onnx.combine(
    onnx_graphs=[
        preprocess_onnx,
        yolov9c_onnx,
        postprocess_onnx,
    ],
    op_prefixes_after_merging=[
        "1",
        "2",
        "3",
    ],
    srcop_destop=[
        [preprocess_onnx.graph.output[0].name, yolov9c_onnx.graph.input[0].name],
        ['2_'+yolov9c_onnx.graph.output[0].name, postprocess_onnx.graph.input[0].name],
    ],
    output_onnx_file_path="onnx_folder/pre_and_yolo_and_post.onnx",
)
pre_and_yolo_and_post = onnx.shape_inference.infer_shapes(pre_and_yolo_and_post)
pre_and_yolo_and_post, check = onnxsim.simplify(pre_and_yolo_and_post)
onnx.save(pre_and_yolo_and_post, "onnx_folder/pre_and_yolo_and_post.onnx")
print("Successfully saved pre_and_yolo_and_post.onnx")

