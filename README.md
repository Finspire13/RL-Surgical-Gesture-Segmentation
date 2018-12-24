# RL-Surgical-Gesture-Segmentation

This is the code of [*Deep Reinforcement Learning for Surgical Gesture Segmentation and Classification*](https://arxiv.org/abs/1806.08089)([Supplementary](https://github.com/Finspire13/RL-Surgical-Gesture-Segmentation/blob/master/supplementary.pdf))

(MICCAI 2018)

This repo includes two parts:
1. Re-implementation of Temporal Convolutional Networks (For the original version, please refer to [Colin Lea's repo](https://github.com/colincsl/TemporalConvolutionalNetworks)).
2. The MDP formulation and a customized OpenAI Gym Env. Policy training is implemented with OpenAI Baseline.

# Install

1. Install scipy, numpy, pandas, matplotlib
2. Install pytorch, tensorflow, gym
3. Clone [this snapshot version of OpenAI Baseline](https://github.com/Finspire13/baselines) and follow its install instructions. [The offical library](https://github.com/openai/baselines) is under rapid development and often brings breaking changes
4. Clone this repo

# Dataset

The code is tested on the three tasks of JIGSAWS dataset (Video and sensor data for the suturing. Only sensor data for the knot-tying and the needle-passing).

We use the same data features and splits as [Colin Lea](https://github.com/colincsl/TemporalConvolutionalNetworks). Request for the data and split files from Colin and put them into folder *./raw_features* and *./splits*.

We also test on other datasets such as 50Salads and GTEA but do not formally benchmark or claim any results.

(Please be careful about the naming of JIGSAWS:

'JIGSAWS': The suturing task in JIGSAWS

'JIGSAWS_K': The knot tying task in JIGSAWS

'JIGSAWS_N': The needle passing task in JIGSAWS)

# How to Run

### Hyper-parameters

Setttings and hyper-parameters are in *config.json*. The defaults work well but you can modify it.

### Run the whole experiment (Recommanded)

First setup the experiment in *experiment.py*. Then:

`cd <this repo>`

`python3 experiment.py`

`python3 export_csv.py` 

The results will be in CSV files. It takes about 24h for one dataset.

### Run TCN alone

`python3 tcn_main.py --feature_type {}`

### Run RL alone (one split)

Train:

`python3 trpo_train.py --feature_type {} --tcn_run_idx {} --split_idx {} --run_idx {}`

Test:

`python3 trpo_test.py --feature_type {} --tcn_run_idx {} --split_idx {} --run_idx {}`

### Output

CSV files: the final results

Folder *result*: the results in npy format

Folder *graph*: plots and figures

Folder *tcn_featrues*: featrues extracted by TCN (states for RL)

Floder *tcn_log*: Training log for TCN (Visulize with tensorboard)

Folder *tcn_model*: Trained models for TCN

Folder *trpo_model*: Trained models for RL


# Contact

Any question please contact with Daochang Liu at finspire13@gmail.com.

# Citation

@InProceedings{10.1007/978-3-030-00937-3_29,
  author="Liu, Daochang and Jiang, Tingting",
  title="Deep Reinforcement Learning for Surgical Gesture Segmentation and Classification",
  booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2018",
  year="2018",
  publisher="Springer International Publishing",
  address="Cham",
  pages="247--255",
  isbn="978-3-030-00937-3"
}
