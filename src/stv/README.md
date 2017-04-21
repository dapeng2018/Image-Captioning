# Skip-Thought Vector* For Guidance Caption Encoding

This directory contains files related to the [Skip-Thought Vector](https://arxiv.org/pdf/1506.06726.pdf) model.
The model is used for generic distributed sentence encoding which is later fed into the attention model as input.
These open-source scripts were written by the TensorFlow authors and are also available through the TensorFlow site and GitHub.

## Files

* [configuration.py](./configuration.py)

    Default configuration for STV model architecture and training.
    - by TensorFlow Authors

* [encoder_manager.py](.encoder_manager.py)

    Management class in charge of the STV model encoder.
    - by TensorFlow Authors

* [gru_cell.py](./gru_cell.py)

    Custom GRU cell modified for the STV model.
    - by TensorFlow Authors

* [input_ops.py](./input_ops.py)

    Contains TensorFlow operations used by the STV model.
    - by TensorFlow Authors

* [special_words.py](./special_words.py)

    Script declaring special tokens used by the STV model as python variables.
    - by TensorFlow Authors

* [skip_thoughts_encoder.py](./skip_thoughts_encoder.py)

    Encoder portion of the Skip-Thought Vector model used to encode the guidance caption.
    - by TensorFlow Authors

* [skip_thoughts_model.py](./skip_thoughts_model.py)

    Skip-Thought vector model (pretrained) based on the [paper.](https://arxiv.org/abs/1506.06726)
    - by TensorFlow Authors