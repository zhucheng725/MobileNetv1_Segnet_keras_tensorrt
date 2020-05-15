

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import time


engine_file = '/home/nvidia/procedure/keras/output/finetuing_mobilenetv1_segnet.engine'
img_path = '/home/nvidia/procedure/keras/JPEGImages/1.jpg'
Input_shape = (224, 224, 3)
DTYPE = trt.float16

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')

def allocate_buffers(engine):
    print('allocate buffers')
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype= trt.nptype(DTYPE))
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype= trt.nptype(DTYPE))		
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    return h_input, d_input, h_output, d_output
		

def do_inference(context, h_input, d_input, h_output, d_output):
    #transfer input data to the GPU
    cuda.memcpy_htod(d_input, h_input)
    #run inference
    context.execute(batch_size = 1, bindings=[int(d_input), int(d_output)])
    #transfer predictions back from GPU
    cuda.memcpy_dtoh(h_output, d_output)
    return h_output


def load_input(img_path, host_buffer):
    h, w, c = Input_shape
    img = cv2.imread(img_path, 1)
    img = cv2.resize(img, (w, h))
    img = img.astype(np.float16)
    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68
    img = img[:, :, ::-1]
    img = img.reshape(-1)
    np.copyto(host_buffer, img)

with open(engine_file, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())


h_input, d_input, h_output, d_output = allocate_buffers(engine)


with engine.create_execution_context() as context:
    start_time = time.time()
    for i in range(10):
        load_input(img_path, host_buffer = h_input)
        output = do_inference(context, h_input, d_input, h_output, d_output)
        print('finished')
    end_time = time.time()

print('use time:', end_time - start_time)




