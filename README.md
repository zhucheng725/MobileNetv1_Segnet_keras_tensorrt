# MobileNetv1_Segnet_keras_tensorrt

# Before Training
I just use VGG16 as backbone, using MobileNetV1 to speed up and decreasing the size of model. Then adding four upsampling layers as final layers can extract segmentation feature. More details about Segnet can be read: https://arxiv.org/pdf/1511.02680.pdf

After training can create a model.h5 files and change our model.h5 file to the model.uff file and change to model.engine. Then you can put the model.engine in Nvidia platform such as NIDIA TX2/ Nano.


Also you can use Pascal VOC datasets to train the whole network without finetuning. Because of my bad tricks, I can not train a good model to be used in outdoor. But when you use finetuning trick can train less time and get a good model to be used.

Some keras pre-train model can be download in these website:
```
https://github.com/fchollet/deep-learning-models/releases
```
And I download the pre-train model named mobilenet_1_0_224_tf_no_top.h5 and use Nano.

```
Jetpack4.4 keras 2.15 tensorflow 1.15 tensorrt 7
```

# Start to train
Put finetuing_mobilenetv1_segnet_voc.py, training.py, voc_reader.py in one file then change images path can be used.

# Change to uff model
To use create_uff.py can change my model.h5 file to a model.uff file.
We can use this code to load my model.h5 file:
```
with CustomObjectScope({'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D, 'relu6':tf.nn.relu6}):
    model = load_model('/home/nvidia/procedure/keras/output/finetuing_mobilenetv1_segnet.h5')
```
More importantly, do not use 
```
'relu6':K.backend.relu6
```
I don't know why can not be used although they look the same but when you change to model.engine, you would meet some errors which can not know relu6.

Then change some nodes and op as follow:
```
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')

dynamic_graph = gs.DynamicGraph(frozen_graph)

add_nodes = dynamic_graph.find_nodes_by_op('AddV2')
for node in add_nodes:
    node.op = 'Add' 

resize_nearest_0 = gs.create_plugin_node('up_sampling2d_1/ResizeNearestNeighbor', op='ResizeNearest_TRT', dtype=tf.float32, scale=2.0)

namespace_plugin_map = {'up_sampling2d_1/ResizeNearestNeighbor': resize_nearest_0,'up_sampling2d_2/ResizeNearestNeighbor': resize_nearest_1,'up_sampling2d_3/ResizeNearestNeighbor':resize_nearest_2,'up_sampling2d_4/ResizeNearestNeighbor':resize_nearest_3}

dynamic_graph.collapse_namespaces(namespace_plugin_map)
new_dynamic_graph = dynamic_graph.as_graph_def()

```

# Change to engine model
To use create_engine_uff.py can change my model.uff file to a model.engine file.
```
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
```

# Start to use engine model
To use use_engine.py can check how fast it is.
I found the ResizeNearest_TRT can not be used in fp16.

![image](https://github.com/zhucheng725/MobileNetv1_Segnet_keras_tensorrt/blob/master/result.jpg)
