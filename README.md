# Dynamic Memory Tensor Networks in Theano
The project is forked from https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano
The aim of this repository is to implement **D**ynamic **M**emory **T**ensor **N**etworks, besides the Dynamic memory networks covered in the parent.

DMTN as described in  https://arxiv.org/abs/1703.03939
*Ramachandran, Govardana Sachithanandam, and Ajay Sohmshetty. "Ask Me Even More: Dynamic Memory Tensor Networks (Extended Model)." arXiv preprint arXiv:1703.03939 (2017).* 

Orginally published as http://cs224d.stanford.edu/reports/SohmshettyRamachandran.pdf
*Sohmshetty, Ajay, and Govardana Sachithanandam Ramachandran. "Ask Me Even More: Dynamic Memory Tensor Networks (Extended Model)." http://cs224d.stanford.edu/reports_2016.html (June 2016)*

**Abstract**:
We examine Memory Networks for the task of question answering (QA), under common real world scenario where training examples are scarce and under weakly supervised scenario, that is only extrinsic labels are available for training. We propose extensions for the Dynamic Memory Network (DMN), specifically within the attention mechanism, we call the resulting Neural Architecture as Dynamic Memory Tensor Network (DMTN). Ultimately, we see that our proposed extensions results in over 80% improvement in the number of task passed against the baselined standard DMN and 20% more task passed compared to state-of-the-art End-to-End Memory Network for Facebook's single task weakly trained 1K bAbi dataset.

![dmtncomparison](https://cloud.githubusercontent.com/assets/19319509/25372789/ac4bd34a-294b-11e7-8455-3ebd26d53c42.jpg)

_[Table:1]Accuracies across all tasks for MemN2N, DMN, and DMTN. Here DMN baselines
serves as the baseline for DTMN to measure the lift with the proposed changes. DMN best* is the
best document performance of DMN with optimal hyperparameter tuning on bAbi weakly trained
dataset- http://yerevann.github.io/2016/02/05/implementing-dynamic-memory-networks_

The above results are obtained by using following Hyper-parameter was used between DMN baseline and DMTN. Please note that due to lack of time & resource Hyper-parameter tunning was not done, Hence we recommend you to play with Hyper-parameter for even better results

![hyper_parameter](https://cloud.githubusercontent.com/assets/19319509/25373388/54ef9f02-294e-11e7-8ed4-9acbdbde9c40.jpg)

_[Table:2]Hyperparameters used for DMN baseline and DMTN._

DMN as described in the [paper by Kumar et al.](http://arxiv.org/abs/1506.07285)
and to experiment with its various extensions.


**Pretrained models on bAbI tasks can be tested [online](http://yerevann.com/dmn-ui/).**

We will cover the process in a series of blog posts.
* [The first post](http://yerevann.github.io/2016/02/05/implementing-dynamic-memory-networks/) describes the details of the basic architecture and presents our first results on [bAbI tasks](http://fb.ai/babi) v1.2.
* [The second post](http://yerevann.github.io/2016/02/23/playground-for-babi-tasks/) describes our second model called `dmn_smooth` and introduces our [playground for bAbI tasks](http://yerevann.com/dmn-ui/).

## Repository contents

| file | description |
| --- | --- |
| `main.py` | the main entry point to train and test available network architectures on bAbI-like tasks |
| `dmn_basic.py` | our baseline implementation. It is as close to the original as we could understand the paper, except the number of steps in the main memory GRU is fixed. Attention module uses `T.abs_` function as a distance between two vectors which causes gradients to become `NaN` randomly.  The results reported in [this blog post](http://yerevann.github.io/2016/02/05/implementing-dynamic-memory-networks/) are based on this network |
| `dmn_smooth.py` | uses the square of the Euclidean distance instead of `abs` in the attention module. Training is very stable. Performance on bAbI is slightly better |
| `dmtn.py` | DMTN implementaion |
| `dmn_batch.py` | `dmn_smooth` with minibatch training support. The batch size cannot be set to `1` because of the [Theano bug](https://github.com/Theano/Theano/issues/1772) | 
| `dmn_qa_draft.py` | draft version of a DMN designed for answering multiple choice questions | 
| `utils.py` | tools for working with bAbI tasks and GloVe vectors |
| `nn_utils.py` | helper functions on top of Theano and Lasagne |
| `fetch_babi_data.sh` | shell script to fetch bAbI tasks (adapted from [MemN2N](https://github.com/npow/MemN2N)) |
| `fetch_glove_data.sh` | shell script to fetch GloVe vectors (by [5vision](https://github.com/5vision/kaggle_allen)) |
| `server/` | contains Flask-based restful api server |


## Usage

This implementation is based on Theano and Lasagne. One way to install them is:

    pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/master/requirements.txt
    pip install https://github.com/Lasagne/Lasagne/archive/master.zip

The following bash scripts will download bAbI tasks and GloVe vectors.

    ./fetch_babi_data.sh
    ./fetch_glove_data.sh

Use `main.py` to train a network:

    python main.py --network dmtn --babi_id 1

The states of the network will be saved in `states/` folder. 
There is one pretrained state on the 1st bAbI task. It should give 100% accuracy on the test set:

    python main.py --network dmtn --mode test --babi_id 1 --load_state states/dmn_basic.mh5.n40.babi1.epoch4.test0.00033.state

### Server

If you want to start a server which will return the predication for bAbi tasks, you should do the following:

1. Generate UI files as described in [YerevaNN/dmn-ui](YerevaNN/dmn-ui)
2. Copy the UI files to `server/ui`
3. Run the server 

```bash
cd server && python api.py
```

If have Docker installed, you can pull our Docker image with ready DMN server.

```bash
docker pull yerevann/docker
docker run --name dmn_1 -it --rm -p 5000:5000 yerevann/dmn
```

## Roadmap

* Mini-batch training ([done](https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano/blob/master/dmn_batch.py), 08/02/2016)
* Web interface ([done](https://github.com/YerevaNN/dmn-ui), 08/23/2016)
* Visualization of episodic memory module ([done](https://github.com/YerevaNN/dmn-ui), 08/23/2016)
* Regularization (work in progress, L2 doesn't help at all, dropout and batch normalization help a little)
* Support for multiple-choice questions ([work in progress](https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano/blob/master/dmn_qa_draft.py))
* Evaluation on more complex datasets
* Import some ideas from [Neural Reasoner](http://arxiv.org/abs/1508.05508)

## License
[The MIT License (MIT)](./LICENSE)
Copyright (c) 2016 YerevaNN
