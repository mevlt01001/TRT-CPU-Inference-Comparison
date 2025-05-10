import onnxruntime
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
import os
import tensorrt
import cv2
from collections import deque

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
    cuda.memset_d32_async(int(output_buffer[1]), 0, output_buffer[0].size,  stream)
    stream.synchronize()   
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
        dummy_input = np.random.randn(720,1280,3).astype(np.int64)
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
    preprocess_session = onnxruntime.InferenceSession("onnx_folder/pre_process.onnx", providers=["CUDAExecutionProvider"])
    if type == "trt":
        yolov9c_engine = load_engine("engine_folder/yolov9c.engine")
        input_buffer, output_buffer, context, stream = allocate_buffers(yolov9c_engine)
        cnt = 0
        start = time.time()
        dummy_input = preprocess_session.run(None, {preprocess_session.get_inputs()[0].name: cv2.VideoCapture(0).read()[1].astype(np.int64)})[0]
        while True:
            cnt +=1
            yolo_output = do_inference(context, stream, input_buffer, output_buffer, dummy_input)
            lat = (time.time() - start)/cnt*1000
            fps = cnt/(time.time() - start)
            print(f"yolov9c_type: {type}, latency: {lat:.2f} ms, fps: {fps:.2f} fps")
    else:
        
        yolov9c_session = onnxruntime.InferenceSession("onnx_folder/yolov9c.onnx", providers=providers[type])        
        dummy_input = np.random.randn(1,3,640,640).astype(np.float32)
        latencies = []
        while True:
            start = time.time()
            yolo_output = yolov9c_session.run(None, {yolov9c_session.get_inputs()[0].name: dummy_input})[0]
            end = time.time()
            latencies.append((end - start))  # Convert to milliseconds
            cnt = len(latencies)-1
            lat = np.mean(latencies[1:-1:1])*1000
            fps = cnt/np.sum(latencies[1:-1:1])
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
        dummy_input = np.random.rand(1,84,8400).astype(np.float32)
        dummy0 = np.random.rand(1,4,8400)
        dummy1 = np.random.rand(1,80,8390)*0.1
        dummy2 = np.random.rand(1,80,10)*0.7+0.3
        dummy3 = np.concatenate((dummy1, dummy2), axis=2)
        dummy_input = np.concatenate((dummy0, dummy3), axis=1).astype(np.float32)
        latencies = deque(maxlen=50)
        while True:
            start = time.time()
            post_output = do_inference(context, stream, input_buffer, output_buffer, dummy_input)
            end = time.time()
            latencies.append(end - start)
            lat = sum(latencies)/len(latencies)
            fps = 1/lat
            print(f"postprocess_type: {type}, latency: {lat*1000:.2f} ms, fps: {fps:.2f} fps")
    else:
        postprocess_session = onnxruntime.InferenceSession("onnx_folder/post_process.onnx", providers=providers[type])
        cnt = 0
        start = time.time()
        dummy_input = np.random.randn(1,84,8400).astype(np.float32)
        dummy_input = np.random.rand(1,84,8400).astype(np.float32)
        dummy0 = np.random.rand(1,4,8400)
        dummy1 = np.random.rand(1,80,8390)*0.1
        dummy2 = np.random.rand(1,80,10)*0.7+0.3
        dummy3 = np.concatenate((dummy1, dummy2), axis=2)
        dummy_input = np.concatenate((dummy0, dummy3), axis=1).astype(np.float32)
        # dummy_input = np.random.randn(1,84,8400).astype(np.float32)
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

def measure_latency_cap(preprocess_type: str, postprocess_type: str, cap: cv2.VideoCapture, yolo=9):
    
    assert preprocess_type in ["cpu", "gpu", "trt"], f"preprocess_type {preprocess_type} not in {['cpu', 'gpu', 'trt']}"
    assert postprocess_type in ["cpu", "gpu", "trt"], f"postprocess_type {postprocess_type} not in {['cpu', 'gpu', 'trt']}"
    assert yolo in [9, 11], f"yolo {yolo} not in {[9, 11]}"
    label = f"preprocess_type: {preprocess_type}, postprocess_type: {postprocess_type}, YOLOv{yolo}"
    providers = {
        "gpu": ["CUDAExecutionProvider"],
        "cpu": ["CPUExecutionProvider"],
    }

    if preprocess_type != "trt" and postprocess_type != "trt":
        # Only YOLOv9c works with engine
        preprocess_session = onnxruntime.InferenceSession("onnx_folder/pre_process.onnx", providers=providers[preprocess_type])
        postprocess_session = onnxruntime.InferenceSession("onnx_folder/post_process.onnx", providers=providers[postprocess_type])

        YOLO_engine = load_engine("engine_folder/yolov9c.engine") if yolo == 9 else load_engine("engine_folder/yolo11m.engine")
        input_buffer, output_buffer, context, stream = allocate_buffers(YOLO_engine)


        lat_buf = deque(maxlen=50)
        while True:

            ret, frame = cap.read()
            start = time.time()
            # Resize frame as required by the model, for example 720p
            pre_output = preprocess_session.run(None, {preprocess_session.get_inputs()[0].name: frame.astype(np.int64)})[0]
            yolo_output = do_inference(context, stream, input_buffer, output_buffer, pre_output)
            post_output = postprocess_session.run(None, {postprocess_session.get_inputs()[0].name: yolo_output})[0]
            end = time.time()
            lat_buf.append(end-start)

            lat = sum(lat_buf)/len(lat_buf)
            fps = 1/(lat)

            for box in post_output:
                x1, y1,x2,y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(frame, f"{label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            info = f"latency: {lat*1000:.2f} ms, fps: {fps:.2f}"
            cv2.putText(frame, info, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    elif preprocess_type == "trt" and postprocess_type != "trt":
        # Preprocess and YOLO works with engine
        postprocess_session = onnxruntime.InferenceSession("onnx_folder/post_process.onnx", providers=providers[postprocess_type])

        pre_and_yolo_engine = load_engine("engine_folder/pre_and_yolo9.engine") if yolo == 9 else load_engine("engine_folder/pre_and_yolo11.engine")
        input_buffer, output_buffer, context, stream = allocate_buffers(pre_and_yolo_engine)

        lat_buf = deque(maxlen=50)
        while True:

            ret, frame = cap.read()
            start = time.time()
            # Resize frame as required by the model, for example 720p
            yolo_output = do_inference(context, stream, input_buffer, output_buffer, frame)
            post_output = postprocess_session.run(None, {postprocess_session.get_inputs()[0].name: yolo_output})[0]
            end = time.time()
            lat_buf.append(end-start)

            lat = sum(lat_buf)/len(lat_buf)
            fps = 1/(lat)

            for box in post_output:
                x1, y1,x2,y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
 
            cv2.putText(frame, f"{label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            info = f"latency: {lat*1000:.2f} ms, fps: {fps:.2f}"
            cv2.putText(frame, info, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    elif preprocess_type != "trt" and postprocess_type == "trt":
        # YOLO and Postprocess works with engine
        preprocess_session = onnxruntime.InferenceSession("onnx_folder/pre_process.onnx", providers=providers[preprocess_type])

        YOLO_and_postprocess_engine = load_engine("engine_folder/yolo9_and_post_process.engine") if yolo == 9 else load_engine("engine_folder/yolo11_and_post_process.engine")
        input_buffer, output_buffer, context, stream = allocate_buffers(YOLO_and_postprocess_engine)

        lat_buf = deque(maxlen=50)
        while True:

            ret, frame = cap.read()
            start = time.time()
            pre_out = preprocess_session.run(None, {preprocess_session.get_inputs()[0].name: frame.astype(np.int64)})[0]
            yolo_output = do_inference(context, stream, input_buffer, output_buffer, pre_out)
            end = time.time()
            lat_buf.append(end-start)

            lat = sum(lat_buf)/len(lat_buf)
            fps = 1/(lat)

            for box in yolo_output:
                x1, y1,x2,y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
 
            cv2.putText(frame, f"{label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            info = f"latency: {lat*1000:.2f} ms, fps: {fps:.2f}"
            cv2.putText(frame, info, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    elif preprocess_type == "trt" and postprocess_type == "trt":
        # Preprocess, YOLOv9c and Postprocess works with engine
        pre_and_yolo_and_postprocess_engine = load_engine("engine_folder/pre_and_yolo_and_post.engine")
        input_buffer, output_buffer, context, stream = allocate_buffers(pre_and_yolo_and_postprocess_engine)

        lat_buf = deque(maxlen=50)
        while True:

            ret, frame = cap.read()
            start = time.time()
            # Resize frame as required by the model, for example 720p
            yolo_output = do_inference(context, stream, input_buffer, output_buffer, frame)
            end = time.time()
            lat_buf.append(end-start)

            lat = sum(lat_buf)/len(lat_buf)
            fps = 1/(lat)

            for box in yolo_output:
                x1, y1,x2,y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
 
            cv2.putText(frame, f"{label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            info = f"latency: {lat*1000:.2f} ms, fps: {fps:.2f}"
            cv2.putText(frame, info, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    else:
        raise ValueError(f"preprocess_type {preprocess_type} and postprocess_type {postprocess_type} not supported")

if __name__ == "__main__":

    # preprocess_type = "trt"  # or "cpu"
    # postprocess_type = "trt"  # or "cpu"
    only_yolov9c_latency("trt")
    # only_preprocess_latency("gpu")
    # only_postprocess_latency("trt")
    # only_cv2_dnn_NMSBoxes_latency()
    # cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    # cap.set(cv2.CAP_PROP_FPS, 60)

    # measure_latency_cap('trt', 'trt', cap=cap, yolo=9)