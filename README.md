# *Image-Captioning* Using Text-Guided Attention

..work in progress

## Prerequisites

* [NLTK](http://www.nltk.org/)
* [NumPy](http://www.numpy.org/)
* [Python 3.5](https://www.python.org/downloads/release/python-350/)
* [scikit-image](http://scikit-image.org/docs/dev/api/skimage.html)
* [TensorFlow](https://www.tensorflow.org/) (>= r1.0)

## Usage

```
python3 test.py 'path/to/input/image' 'path/to/trained/model'
```

```
python3 train.py
```

## Files

* [atention.py](./src/attention.py)

    Class responsible for text-guided attention.

* [caption_extractor.py](./src/caption_extractor.py)

    Class responsible extracting a guidance caption and other operations related to ETL of sentences.

* [cider.py](./src/cider/cider.py)

    Describes the class to compute the CIDEr (Consensus-Based Image Description Evaluation) Metric 
    
    - by Vedantam, Zitnick, and Parikh (http://arxiv.org/abs/1411.5726)


* [cider_scorer.py](./src/cider/cider_scorer.py)

    Contains the actual methods for computing the cider score.
    
    - by Vedantam, Zitnick, and Parikh


* [configuration.py](./src/stv/configuration.py)

    Default configuration for STV model architecture and training.
    - by TensorFlow Authors
    
* [decoder.py](./src/decoder.py)

    Class responsible for decoding a context vector into a predicted encoded caption.

* [encoder_manager.py](./src/stv/encoder_manager.py)

    - by TensorFlow Authors

* [fcn16_vgg.py](./src/vgg/fcn16_vgg.py)

    VGG network that is fine-tuned on the MS-COCO dataset to predict image attributes.

* [gru_cell.py](./src/stv/gru_cell.py)

    - by TensorFlow Authors

* [helpers.py](./src/helpers.py)

    Contains various auxiliary functions.

* [input_ops.py](./src/stv/input_ops.py)

    - by TensorFlow Authors
 
* [neighbor.py](./src/neighbor.py)

    This class is simply responsible for determining the nearest neighboring images of an image.
 
* [setup.sh](./bin/setup.sh) 

    Executable shell script for downloading, extrcating, and installing prerequisite files.

* [special_words.py](./src/stv/special_words.py)

    - by TensorFlow Authors

* [skip_thoughts_encoder.py](./src/stv/skip_thoughts_encoder.py)

    - by TensorFlow Authors

* [skip_thoughts_model.py](./src/stv/skip_thoughts_model.py)
 
    - by TensorFlow Authors
 
* [test.py](./src/test.py)

    Executable script for generating a caption given an input image and an already trained model.

* [train.py](./src/train.py)

    Executable script for training a new captioning model.
    
* [vocab.py](./src/vocab.py)

    Class responsible for vocabulary retrieval, encoding, decoding, and other operations.

