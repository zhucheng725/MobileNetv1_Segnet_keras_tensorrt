import tensorrt as trt

model_file = '/home/nvidia/procedure/keras/output/finetuing_mobilenetv1_segnet.uff'

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')

def build_engine(model_file):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        builder.max_workspace_size = 1 << 20
        builder.max_batch_size = 1
        #builder.fp16_mode = True
        #builder.strict_type_constraints =  True
        parser.register_input("input_1", (224,224,3),trt.UffInputOrder.NHWC)
        parser.register_output("activation_1/truediv")
        parser.parse(model_file, network)
        return builder.build_cuda_engine(network)


with build_engine(model_file) as engine:
    print('1')
    with open('/home/nvidia/procedure/keras/output/finetuing_mobilenetv1_segnet.engine','wb') as f:
        f.write(engine.serialize())
