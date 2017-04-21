# *CIDEr Scoring* For Guidance Caption Extraction

This directory contains files related to [Consensus-Based Image Description Evaluation](https://arxiv.org/pdf/1411.5726.pdf) scoring.
Scores are computed to measure similarities amongst a set of candidate captions in order to extract a guidance caption.
These open-source scripts written by the original authors of the paper can be found on one of their [GitHub repositories](https://github.com/vrama91/cider).

## Files

* [cider.py](./cider.py)

    Describes the class to compute the CIDEr Metric
    - by Vedantam, Zitnick, and Parikh

* [cider_scorer.py](./cider_scorer.py)

    Contains the actual methods for computing the cider score.
    - by Vedantam, Zitnick, and Parikh
