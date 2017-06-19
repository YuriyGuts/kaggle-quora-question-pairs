# kaggle-quora-question-pairs

My solution to [Kaggle Quora Question Pairs competition](https://www.kaggle.com/c/quora-question-pairs) (Top 2%, Private LB log loss 0.13497).



## Overview

The solution uses a mixture of purely statistical features, classical NLP features, and deep learning.
Almost 200 handcrafted features are combined with out-of-fold predictions from 4 neural networks having different architectures.

The final model is a GBM (LightGBM), trained with early stopping and a very small learning rate, using stratified K-fold cross validation.

![Overall solution structure](assets/solution-diagram.png)


## Reproducing the Solution


### Hardware Requirements

Almost all notebooks (with the exception of some 3rd-party scripts) can efficiently utilize multi-core machines.
At the same time, some of them might be memory-hungry.
All code has been tested on a machine with 64 GB RAM.
For all non-neural notebooks, a `c4.8xlarge` AWS instance should do excellent.

For neural networks, a GPU is highly recommended.
On a GTX 1080 Ti, it takes about 8-9 hours to complete all 4 "neural" notebooks.


### Software Requirements

1. Python >= 3.6.
2. LightGBM (compiled from sources).
3. FastText (compiled from sources).
4. Python packages from `requirements.txt`.
5. (Recommended) NVIDIA CUDA and a GPU version of TensorFlow.


### Environment Provisioning

(WiP) You can spin up a fresh Ubuntu 16.04 AWS instance and use Ansible to make all the necessary software installation and configuration (except CUDA/Tensorflow).

1. Navigate to the `provisioning` directory.
2. Edit `inventory.ini` and specify your instance DNS and the private key to access it.
3. Run:
    ```
    $ ansible-galaxy install -r requirements.yml
    $ ansible-playbook playbook.yml -i inventory.ini
    ``` 

### Running the Code

Start a Jupyter server in the `notebooks` directory. 
Run the notebooks in the following order:

1. **Preprocessing**.
    ```
    1) preproc-tokenize-spellcheck.ipynb
    2) preproc-extract-unique-questions.ipynb
    3) preproc-embeddings-fasttext.ipynb
    4) preproc-nn-sequences-fasttext.ipynb
    ```

2. **Feature extraction**.

    Run all `feature-*.ipynb` notebooks in arbitrary order.
    
    *Note*: for faster execution, run all `feature-oofp-nn-*.ipynb` notebooks on a machine with a GPU and NVIDIA CUDA.
    
3. **Prediction**.

    Run `classify-lightgbm-cv-pred.ipynb`.
    The output file will be saved as `DATETIME-submission-draft-CVSCORE.csv`
