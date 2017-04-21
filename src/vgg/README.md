# *Fully Connected VGG16* For Image Encoding

This directory contains the file related to the VGG convolutional neural network of the 16 layer variety.
The network model has been specifically fine-tuned to predict image attributes.
It is used to encode images for deriving guidance captions and to use as input to the attention model.

## Files

* [fcn16_vgg.py](./src/vgg/fcn16_vgg.py)

    VGG network that is fine-tuned on the MS-COCO dataset to predict image attributes.