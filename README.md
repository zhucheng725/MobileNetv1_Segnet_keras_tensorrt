# MobileNetv1_Segnet_keras_tensorrt

# Before Training
I just use VGG16 as backbone, using MobileNetV1 to speed up and decreasing the size of model. Then adding four upsampling layers as final layers can extract segmentation feature. More details about Segnet can be read: https://arxiv.org/pdf/1511.02680.pdf

After training can create a model.h5 files and change our model.h5 file to the model.uff file and change to model.engine. Then you can put the model.engine in Nvidia platform such as NIDIA TX2/ Nano.


Also you can use Pascal VOC datasets to train the whole network without finetuning. Because of my bad tricks, I can not train a good model to be used in outdoor. But when you use finetuning trick can train less time and get a good model to be used.

Some keras pre-train model can be download in these website:
```
https://github.com/fchollet/deep-learning-models/releases
```
And I download the pre-train model named mobilenet_1_0_224_tf_no_top.h5

# Start to train
Put finetuing_mobilenetv1_segnet_voc.py, training.py, voc_reader.py in one file then change images path can be used.

# Change to uff model


