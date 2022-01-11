### This is the online repository of the ESEC/FSE2021 paper titled "Lightweight Global and Local Contexts Guided Method Name Recommendation with Prior Knowledge".

# Datasets

All datasets used in our study are open-sourced. We provide the links to each of them below.

* Empirical dataset ([here](http://groups.inf.ed.ac.uk/cup/javaGithub/))
* MNR task datasets: *Java-small, Java-med, Java-large* ([here](https://github.com/tech-srl/code2seq))
* MNR task dataset: *MNire's* ([here](https://sonvnguyen.github.io/mnire/test_2.zip))
* MCC task dataset ([here](https://github.com/SerVal-DTF/debug-method-name))

# Source Code

## Requirements

Our ***Cognac*** is implemented by following the PyTorch version of [pointer generator network](https://github.com/atulkum/pointer_summarizer).
It is built on PyTorch-1.5 and TensorFlow-1.12.
We use FastText to embed each token and utilize the Python package javalang to perform program analysis. Link to the installation of this package is [here](https://github.com/Ringbo/javalang).

## Reproduction Steps

To reproduce our study, you need to:

1. Execute `dataextractor.py` to extract the inputs of ***Cognac***;
2. Execute `train_fasttext.py` to train the FastText model with using the extracted data from the last step.
3. Train, validate, and test the model by executing `start_train.sh`, `start_eval.sh`, and `start_decode.sh` respectively.
4. If you want to reproduce the MCC task, execute `decode_mcc.py` and `cal_sim.py` respectively.

## Performance Analysis

We are unsure that other reproduction studies can achieve the same results as ours.
Reasons for such deviation can come from:

1. The hyperparameters in the `config.py` file may need to be fine-tuned.
2. In `datasetextractor.py`, we set a threshold to restrict the time consumption for parsing each Java file. Hence, servers with different hardware configuration may parse diverse numbers of methods.
