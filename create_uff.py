
from keras.models import load_model
import cv2
import time
import numpy as np 
import tensorflow as tf
import keras.backend as K
from keras.backend.tensorflow_backend import set_session

import uff
import graphsurgeon as gs
import tensorrt as trt
from keras.utils.generic_utils import CustomObjectScope
import keras


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.log_device_placement = True
sess =  tf.Session(config = config)
set_session(sess)
keras.backend.get_session().run(tf.initialize_all_variables())



K.set_learning_phase(0)


with CustomObjectScope({'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D, 'relu6':tf.nn.relu6}):
    model = load_model('/home/nvidia/procedure/keras/output/finetuing_mobilenetv1_segnet.h5')

model.summary()
K.set_learning_phase(0)

output_name = model.output.op.name
input_name = model.input.op.name


frozen_graph = tf.graph_util.remove_training_nodes(tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(),[output_name]))


TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')

dynamic_graph = gs.DynamicGraph(frozen_graph)

add_nodes = dynamic_graph.find_nodes_by_op('AddV2')
for node in add_nodes:
    node.op = 'Add' 

resize_nearest_0 = gs.create_plugin_node('up_sampling2d_1/ResizeNearestNeighbor', op='ResizeNearest_TRT', dtype=tf.float32, scale=2.0)
resize_nearest_1 = gs.create_plugin_node('up_sampling2d_2/ResizeNearestNeighbor', op='ResizeNearest_TRT', dtype=tf.float32, scale=2.0)
resize_nearest_2 = gs.create_plugin_node('up_sampling2d_3/ResizeNearestNeighbor', op='ResizeNearest_TRT', dtype=tf.float32, scale=2.0)
resize_nearest_3 = gs.create_plugin_node('up_sampling2d_4/ResizeNearestNeighbor', op='ResizeNearest_TRT', dtype=tf.float32, scale=2.0)


namespace_plugin_map = {'up_sampling2d_1/ResizeNearestNeighbor': resize_nearest_0,'up_sampling2d_2/ResizeNearestNeighbor': resize_nearest_1,'up_sampling2d_3/ResizeNearestNeighbor':resize_nearest_2,'up_sampling2d_4/ResizeNearestNeighbor':resize_nearest_3}

dynamic_graph.collapse_namespaces(namespace_plugin_map)

new_dynamic_graph = dynamic_graph.as_graph_def()


uff_model = uff.from_tensorflow(new_dynamic_graph, output_filename = '/home/nvidia/procedure/keras/output/finetuing_mobilenetv1_segnet.uff')



