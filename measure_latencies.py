import onnxruntime
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
import os
import tensorrt
import cv2


def load_engine(engine_path: str):
    """
    Load the TensorRT engine from the specified path.
    """
    with open(engine_path, "rb") as f:
        engine = f.read()
    runtime = tensorrt.Runtime(tensorrt.Logger(tensorrt.Logger.INFO))
    if engine is None:
        raise RuntimeError(f"Failed to load engine from {engine_path}")
    print(f"Successfully loaded engine from {engine_path}")
    return runtime.deserialize_cuda_engine(engine)

def allocate_buffers(engine=None):
    assert engine is not None, "engine must be provided"

    input_tensor = engine.get_tensor_name(0)
    input_shape = engine.get_tensor_shape(input_tensor)
    input_dtype = tensorrt.nptype(engine.get_tensor_dtype(input_tensor))
    # input_like = np.random.randn(*input_shape).astype(input_dtype)
    
    output_tensor = engine.get_tensor_name(1)
    output_shape = engine.get_tensor_shape(output_tensor)
    if -1 in output_shape:
        output_shape = (100,4)
    output_dtype = tensorrt.nptype(engine.get_tensor_dtype(output_tensor))
    # output_like = np.random.randn(*output_shape).astype(output_dtype)

    info=f"""
    input_tensor: {input_tensor}
    input_shape: {input_shape}
    input_dtype: {input_dtype}
    output_tensor: {output_tensor}
    output_shape: {output_shape}
    output_dtype: {output_dtype}
    """
    print(info)
    input_like = np.random.randn(*input_shape).astype(input_dtype)
    output_like = np.random.randn(*output_shape).astype(output_dtype)

    CPU_INPUT_BUFFER = cuda.pagelocked_empty_like(input_like)
    CPU_OUTPUT_BUFFER = cuda.pagelocked_empty_like(output_like)
    GPU_INPUT_BUFFER = cuda.mem_alloc_like(CPU_INPUT_BUFFER)
    GPU_OUTPUT_BUFFER = cuda.mem_alloc_like(CPU_OUTPUT_BUFFER)

    context = engine.create_execution_context()
    context.set_tensor_address(engine.get_tensor_name(0), int(GPU_INPUT_BUFFER))
    context.set_tensor_address(engine.get_tensor_name(1), int(GPU_OUTPUT_BUFFER))
    stream = cuda.Stream()
    
    return [CPU_INPUT_BUFFER, GPU_INPUT_BUFFER], [CPU_OUTPUT_BUFFER, GPU_OUTPUT_BUFFER], context, stream

def do_inference(context, stream, input_buffer, output_buffer, data):
    """
    Run inference on the input buffer and return the output buffer.
    """
    input_buffer[0][:] = data
    cuda.memcpy_htod_async(input_buffer[1], input_buffer[0], stream)
    context.execute_async_v3(stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(output_buffer[0], output_buffer[1], stream)
    return output_buffer[0]

def measure_latency(preprocess_type: str, postprocess_type: str):
    """
    default onnx path: onnx_folder/\n
    default engine path: engine_folder/
    """
    providers = {
        "gpu": ["CUDAExecutionProvider"],
        "cpu": ["CPUExecutionProvider"],
    }

    assert preprocess_type in ["cpu", "gpu", "trt"], f"preprocess_type {preprocess_type} not in {['cpu', 'gpu', 'trt']}"
    assert postprocess_type in ["cpu", "gpu", "trt"], f"postprocess_type {postprocess_type} not in {['cpu', 'gpu', 'trt']}"



    if preprocess_type != "trt" and postprocess_type != "trt":
        # Only YOLOv9c works with engine
        preprocess_session = onnxruntime.InferenceSession("onnx_folder/pre_process.onnx", providers=providers[preprocess_type])
        postprocess_session = onnxruntime.InferenceSession("onnx_folder/post_process.onnx", providers=providers[postprocess_type])

        YOLOv9c_engine = load_engine("engine_folder/yolov9c.engine")
        input_buffer, output_buffer, context, stream = allocate_buffers(YOLOv9c_engine)

        cnt = 0
        start = time.time()
        dummy_input = np.random.randn(720,1280,3).astype(np.float32)
        while True:
            cnt +=1
            pre_output = preprocess_session.run(None, {preprocess_session.get_inputs()[0].name: dummy_input})[0]
            yolo_ouput = do_inference(context, stream, input_buffer, output_buffer, pre_output)
            post_output = postprocess_session.run(None, {postprocess_session.get_inputs()[0].name: yolo_ouput})[0]
            lat = (time.time() - start)/cnt*1000
            fps = cnt/(time.time() - start)
            print(f"preprocess_type: {preprocess_type}, postprocess_type: {postprocess_type}, latency: {lat:.2f} ms, fps: {fps:.2f} fps")
    elif preprocess_type == "trt" and postprocess_type != "trt":
        # Preprocess and YOLOv9c works with engine
        postprocess_session = onnxruntime.InferenceSession("onnx_folder/post_process.onnx", providers=providers[postprocess_type])

        pre_and_yolo_engine = load_engine("engine_folder/pre_and_yolo.engine")
        input_buffer, output_buffer, context, stream = allocate_buffers(pre_and_yolo_engine)

        cnt = 0
        start = time.time()
        dummy_input = np.random.randn(720,1280,3).astype(np.float32)
        while True:
            cnt +=1
            pre_and_yolo_output = do_inference(context, stream, input_buffer, output_buffer, dummy_input)
            post_output = postprocess_session.run(None, {postprocess_session.get_inputs()[0].name: pre_and_yolo_output})[0]
            lat = (time.time() - start)/cnt*1000
            fps = cnt/(time.time() - start)
            print(f"preprocess_type: {preprocess_type}, postprocess_type: {postprocess_type}, latency: {lat:.2f} ms, fps: {fps:.2f} fps")
    
    elif preprocess_type != "trt" and postprocess_type == "trt":
        # YOLOv9c and Postprocess works with engine
        preprocess_session = onnxruntime.InferenceSession("onnx_folder/pre_process.onnx", providers=providers[preprocess_type])

        YOLOv9c_and_postprocess_engine = load_engine("engine_folder/yolo_and_post_process.engine")
        input_buffer, output_buffer, context, stream = allocate_buffers(YOLOv9c_and_postprocess_engine)

        cnt = 0
        start = time.time()
        dummy_input = np.random.randn(720,1280,3).astype(np.float32)
        while True:
            cnt +=1
            pre_output = preprocess_session.run(None, {preprocess_session.get_inputs()[0].name: dummy_input})[0]
            yolo_and_postprocess_output = do_inference(context, stream, input_buffer, output_buffer, pre_output)
            lat = (time.time() - start)/cnt*1000
            fps = cnt/(time.time() - start)
            print(f"preprocess_type: {preprocess_type}, postprocess_type: {postprocess_type}, latency: {lat:.2f} ms, fps: {fps:.2f} fps")
    elif preprocess_type == "trt" and postprocess_type == "trt":
        # Preprocess, YOLOv9c and Postprocess works with engine
        pre_and_yolo_and_postprocess_engine = load_engine("engine_folder/pre_and_yolo_and_post.engine")
        input_buffer, output_buffer, context, stream = allocate_buffers(pre_and_yolo_and_postprocess_engine)

        cnt = 0
        start = time.time()
        dummy_input = np.random.randn(720,1280,3).astype(np.float32)
        while True:
            cnt +=1
            pre_and_yolo_and_postprocess_output = do_inference(context, stream, input_buffer, output_buffer, dummy_input)
            lat = (time.time() - start)/cnt*1000
            fps = cnt/(time.time() - start)
            print(f"preprocess_type: {preprocess_type}, postprocess_type: {postprocess_type}, latency: {lat:.2f} ms, fps: {fps:.2f} fps")
    else:
        raise ValueError(f"preprocess_type {preprocess_type} and postprocess_type {postprocess_type} not supported")
    

def only_preprocess_latency(type: str):
    assert type in ["cpu", "gpu", "trt"], f"type {type} not in {['cpu', 'gpu', 'trt']}"
    providers = {
        "gpu": ["CUDAExecutionProvider"],
        "cpu": ["CPUExecutionProvider"],
    }
    if type == "trt":
        preprocess_engine = load_engine("engine_folder/pre_process.engine")
        input_buffer, output_buffer, context, stream = allocate_buffers(preprocess_engine)
        cnt = 0
        start = time.time()
        dummy_input = np.random.randn(720,1280,3).astype(np.float32)
        while True:
            cnt +=1
            pre_output = do_inference(context, stream, input_buffer, output_buffer, dummy_input)
            lat = (time.time() - start)/cnt*1000
            fps = cnt/(time.time() - start)
            print(f"preprocess_type: {type}, latency: {lat:.2f} ms, fps: {fps:.2f} fps")
    else:
        preprocess_session = onnxruntime.InferenceSession("onnx_folder/pre_process.onnx", providers=providers[type])
        cnt = 0
        start = time.time()
        dummy_input = np.random.randn(720,1280,3).astype(np.float32)
        while True:
            cnt +=1
            pre_output = preprocess_session.run(None, {preprocess_session.get_inputs()[0].name: dummy_input})[0]
            lat = (time.time() - start)/cnt*1000
            fps = cnt/(time.time() - start)
            print(f"preprocess_type: {type}, latency: {lat:.2f} ms, fps: {fps:.2f} fps")

def only_yolov9c_latency(type: str):
    assert type in ["cpu", "gpu", "trt"], f"type {type} not in {['cpu', 'gpu', 'trt']}"
    providers = {
        "gpu": ["CUDAExecutionProvider"],
        "cpu": ["CPUExecutionProvider"],
    }
    if type == "trt":
        yolov9c_engine = load_engine("engine_folder/yolov9c.engine")
        input_buffer, output_buffer, context, stream = allocate_buffers(yolov9c_engine)
        cnt = 0
        start = time.time()
        dummy_input = np.random.randn(1,3,640,640).astype(np.float32)
        while True:
            cnt +=1
            yolo_output = do_inference(context, stream, input_buffer, output_buffer, dummy_input)
            lat = (time.time() - start)/cnt*1000
            fps = cnt/(time.time() - start)
            print(f"yolov9c_type: {type}, latency: {lat:.2f} ms, fps: {fps:.2f} fps")
    else:
        yolov9c_session = onnxruntime.InferenceSession("onnx_folder/yolov9c.onnx", providers=providers[type])
        cnt = 0
        start = time.time()
        dummy_input = np.random.randn(1,3,640,640).astype(np.float32)
        while True:
            cnt +=1
            yolo_output = yolov9c_session.run(None, {yolov9c_session.get_inputs()[0].name: dummy_input})[0]
            lat = (time.time() - start)/cnt*1000
            fps = cnt/(time.time() - start)
            print(f"yolov9c_type: {type}, latency: {lat:.2f} ms, fps: {fps:.2f} fps")

def only_postprocess_latency(type: str):
    assert type in ["cpu", "gpu", "trt"], f"type {type} not in {['cpu', 'gpu', 'trt']}"
    providers = {
        "gpu": ["CUDAExecutionProvider"],
        "cpu": ["CPUExecutionProvider"],
    }
    if type == "trt":
        postprocess_engine = load_engine("engine_folder/post_process.engine")
        input_buffer, output_buffer, context, stream = allocate_buffers(postprocess_engine)
        cnt = 0
        start = time.time()
        dummy_input = np.random.randn(1,84,8400).astype(np.float32)
        while True:
            cnt +=1
            post_output = do_inference(context, stream, input_buffer, output_buffer, dummy_input)
            lat = (time.time() - start)/cnt*1000
            fps = cnt/(time.time() - start)
            print(f"postprocess_type: {type}, latency: {lat:.2f} ms, fps: {fps:.2f} fps")
    else:
        postprocess_session = onnxruntime.InferenceSession("onnx_folder/post_process.onnx", providers=providers[type])
        cnt = 0
        start = time.time()
        dummy_input = np.random.randn(1,84,8400).astype(np.float32)
        while True:
            cnt +=1
            post_output = postprocess_session.run(None, {postprocess_session.get_inputs()[0].name: dummy_input})[0]
            lat = (time.time() - start)/cnt*1000
            fps = cnt/(time.time() - start)
            print(f"postprocess_type: {type}, latency: {lat:.2f} ms, fps: {fps:.2f} fps")

def only_cv2_dnn_NMSBoxes_latency():
    cnt = 0
    start = time.time()
    dummy_input = np.random.randn(1, 84, 8400).astype(np.float32)

    while True:
        cnt += 1
        dummy_input = dummy_input.transpose(0, 2, 1) 
        boxes = dummy_input[0][:, :4] 
        scores = dummy_input[0][:, 4:]

        class_indices = np.argmax(scores, axis=1)
        selected_scores = scores[np.arange(scores.shape[0]), class_indices]  

        bboxes = boxes
        scores_list = selected_scores

        indices = cv2.dnn.NMSBoxes(bboxes, scores_list, 0.5, 0.5)

        bboxes = bboxes[indices]
        lat = (time.time() - start) / cnt * 1000
        fps = cnt / (time.time() - start)
        print(f"postprocess_type: cv2.dnn.NMSBoxes, latency: {lat:.2f} ms, fps: {fps:.2f} fps")

if __name__ == "__main__":

    # preprocess_type = "trt"  # or "cpu"
    # postprocess_type = "trt"  # or "cpu"
    # measure_latency(preprocess_type, postprocess_type)
    # only_yolov9c_latency("trt")
    # only_preprocess_latency("trt")
    # only_postprocess_latency("trt")
    only_cv2_dnn_NMSBoxes_latency()


