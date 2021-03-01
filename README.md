# RoR

This repo provides the code for reproducing the experiments in RoR: Read-over-Read for Concersational Mechaine Reading Comprehensition task (C-MRC). In this paper, we propose a Read-over-Read pipeline for tackle the problem of long input in C-MRC task. 

## Dependency

- pip install tensorflow-gpu=1.13

## Pre-trained Models

RoR uses the pre-train model [ELECTRA-large](https://github.com/google-research/electra) as the text encoder. You should download it and put the ckpt models and config.json under the of model. 

## Dataset

The training set and validation set of QuAC are [here](https://quac.ai/).

## Run

sh run.sh

run.sh is the script of RoR pipeline.
