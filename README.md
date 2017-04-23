# *Image-Captioning* Using Text-Guided Attention

<img src="lib/examples/cover.jpg" height="480px" width="640px" align="right">

This is a TensorFlow implementation of *[Text-guided Attention Model for Image Captioning](https://arxiv.org/pdf/1612.03557.pdf)* using [scheduled sampling](https://arxiv.org/pdf/1612.03557.pdf) as a learning approach.
An attention machanism is used by observing associated captions and steering visual attention. 
The model implicitly learns not only what objects look like based on words, but the context of an image, allowing it to naturally describe the scene.

It makes use of a fine tuned [VGG convolutional neural network](https://arxiv.org/pdf/1409.1556.pdf) and a [Skip-Thought Vector](https://arxiv.org/pdf/1506.06726.pdf) model to encode images and captions respectively.
Attention is conducted on these encodings in order to produce a context vector, which is then deciphered by an LSTM into a readable caption.

#### Implementation Architecture

Given some image as input, an image and caption encoding are computed through two independent intermediary models.

For the image encoding, the final fully connected layer of a pretrained VGG network of the 16 layer variety is used. 
This model has been specifically fine-tuned to predict image attributes.

For the caption encoding, the last hidden state (GRU) of a pretrained STV model is used.
The input caption fed into this RNN is chosen through an intricate caption extraction process.
From the input image, a set of candidate captions are initially retrieved based on the top *n* visually similar images.
The top *k* candidates are filtered from the list based on computed [CIDEr](https://arxiv.org/pdf/1411.5726.pdf) scores.
While the top caption is simply chosen during inference, a random caption is chosen from these top captions during training to prevent overfitting.
As mentioned prior, this selected captionc alled the guidance caption is fed into the STV and encoded.

A text-guided attention model then takes as input the image and caption encodings. 
It computes a context vector through an attention mechanism by "attending" different parts (filter maps) of the image encoding at each feature representation of the caption encoding.

Finally, a decoder takes the context vector as input and generates a caption.
It is comprised of a word embedding, LSTM, and prediction layer.
The context vector is first mapped onto a word vector and is used as the initial word.
A new word is predicted at each time step until "\<eos>" is reached.
Dropout is used on the output layer of the decoder.

## Prerequisites

* [NLTK](http://www.nltk.org/)
* [NumPy](http://www.numpy.org/)
* [Python 3.5](https://www.python.org/downloads/release/python-350/)
* [scikit-image](http://scikit-image.org/docs/dev/api/skimage.html)
* [TensorFlow](https://www.tensorflow.org/) (>= r1.0)

## Usage

To generative a caption for a given image, invoke *[test.py](./src/test.py)* with its file path through --input and specify a trained model through --model_path (i.e., "model.ckpt").

```
python3 test.py 'path/to/input/image' 'path/to/trained/model.ckpt'
```

To train a new generative model to caption images invoke *[train.py](./src/train.py)* without supplying additional flags. 
Checkpoints will be generated on occasion during the training session under [lib/models](.lib/models) prefixed with a timestamp for a name.
On completion, a final model will be saved under the same directory but with the additional prefix "trained" before the timestamp.

```
python3 train.py
```

## Files

* [attention.py](./src/attention.py)

    Class responsible for text-guided attention.

* [caption_extractor.py](./src/caption_extractor.py)

    Class responsible extracting a guidance caption and other operations related to ETL of sentences.
    
* [decoder.py](./src/decoder.py)

    Class responsible for decoding a context vector into a predicted encoded caption.

* [helpers.py](./src/helpers.py)

    Contains various auxiliary functions.
 
* [neighbor.py](./src/neighbor.py)

    This class is simply responsible for determining the nearest neighboring images of an image.
 
* [setup.sh](./bin/setup.sh) 

    Executable shell script for downloading, extrcating, and installing prerequisite files.
 
* [test.py](./src/test.py)

    Executable script for generating a caption given an input image and an already trained model.

* [train.py](./src/train.py)

    Executable script for training a new captioning model.
    
* [vocab.py](./src/vocab.py)

    Class responsible for vocabulary retrieval, encoding, decoding, and other operations.

