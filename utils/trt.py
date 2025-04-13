import tensorrt
import numpy as np
import pycuda.driver as cuda

def load_engine(engine_file):
    with open(engine_file, "rb") as f:
        engine = f.read()

    LOGGER = tensorrt.Logger(tensorrt.Logger.INFO)
    RUNTIME = tensorrt.Runtime(LOGGER)

    ICudaEngine = RUNTIME.deserialize_cuda_engine(engine)
    print(f"{'Engine file couldnt deserialized!!' if ICudaEngine is None else 'Engine file has deserialized successfuly!!'}")

    return ICudaEngine

class TensorHostDeviceBuffer:
    def __init__(self, name, cpu_buffer, gpu_buffer):
        self.name = name
        self.cpu_buffer = cpu_buffer
        self.gpu_buffer = gpu_buffer

def allocate_buffers(engine, outshape=None):
    input_cpu_gpu_buffers = []
    output_cpu_gpu_buffers = []

    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        tensor_type = engine.get_tensor_dtype(tensor_name)
        tensor_shape = engine.get_tensor_shape(tensor_name) if engine.get_tensor_shape(tensor_name)[0] != -1 else outshape
        tensor_dtype = tensorrt.nptype(tensor_type)
        tensor_size = tensorrt.volume(tensor_shape)

        cpu_memory = cuda.pagelocked_empty(tensor_size, tensor_dtype)
        gpu_memory = cuda.mem_alloc(cpu_memory.nbytes)
        if engine.get_tensor_mode(tensor_name) == tensorrt.TensorIOMode.INPUT:
            input_cpu_gpu_buffers.append(TensorHostDeviceBuffer(tensor_name, cpu_memory, gpu_memory))
        else:
            output_cpu_gpu_buffers.append(TensorHostDeviceBuffer(tensor_name, cpu_memory, gpu_memory))

    return input_cpu_gpu_buffers, output_cpu_gpu_buffers

def create_execution_context(engine, input_buffers, output_buffers):
    context = engine.create_execution_context()
    for buffer in input_buffers:
        context.set_tensor_address(buffer.name, buffer.gpu_buffer)
    for buffer in output_buffers:
        context.set_tensor_address(buffer.name, buffer.gpu_buffer)
    return context, cuda.Stream()

def run_inference(context, stream, input_buffers, output_buffers, data):
    for buffer in input_buffers:
        np.copyto(buffer.cpu_buffer, data.ravel())
        cuda.memcpy_htod_async(buffer.gpu_buffer, buffer.cpu_buffer, stream)

    context.execute_async_v3(stream_handle=stream.handle)
    
    for buffer in output_buffers:
        cuda.memcpy_dtoh_async(buffer.cpu_buffer, buffer.gpu_buffer, stream)
    
    # stream.synchronize() # yavaşlatıyor
    
    return [buffer.cpu_buffer for buffer in output_buffers]