# RoR

This repo provides the code for reproducing the experiments of RoR in QuAC dataset. 

## Environment

python == 3.6.7 <br>
tensorflow-gpu == 1.13

## Pre-trained Models

RoR uses the pre-trained model [ELECTRA-large](https://github.com/google-research/electra) as the text encoder. You should download it and put the ckpt models and config.json under `$Model_DIR`. 

## Dataset

The training set and validation set of QuAC are [here](https://quac.ai/).

## Running

Experiment configuration

```
GPUDEVICE='0'
NUMGPUS=1
TASKNAME='quac'
RANDOMSEED=100
PREDICTTAG=best
MODELDIR='./model'
DATADIR='./quac'
OUTPUTDIR='./output'
NUMTURN=2
SEQLEN=512
QUERYLEN=128
ANSWERLEN=64
BATCHSIZE=4
LEARNINGRATE=2e-5
TRAINSTEPS=30000
WARMUPSTEPS=0
SAVESTEPS=1000
ANSWERTHRESHOLD=0.3
DOLOWERCASE=true
LAYERDECAY=0.75
MIDDLELOSS=false
DECAYMETHOD='cos'
ADJUSTLOSS=true
LABELSMOOTHING=false
RERANK=false
FREEZEBASELINE=false
```

First reading (chunk reader)
```
CUDA_VISIBLE_DEVICES=${GPUDEVICE} python3 electra_quac.py \
--vocab_file=${MODELDIR}/vocab.txt \
--model_config_path=${MODELDIR}/bert_config.json \
--init_checkpoint=${MODELDIR}/model.ckpt-20000 \
--task_name=${TASKNAME} \
--random_seed=${RANDOMSEED} \
--predict_tag=${PREDICTTAG} \
--do_lower_case=${DOLOWERCASE} \
--data_dir=${DATADIR}/ \
--output_dir=${OUTPUTDIR}/data \
--model_dir=${OUTPUTDIR}/checkpoint  \
--export_dir=${OUTPUTDIR}/export \
--num_turn=${NUMTURN} \
--max_seq_length=${SEQLEN} \
--max_query_length=${QUERYLEN} \
--max_answer_length=${ANSWERLEN} \
--train_batch_size=${BATCHSIZE} \
--predict_batch_size=${BATCHSIZE} \
--num_hosts=1 \
--num_core_per_host=1 \
--learning_rate=${LEARNINGRATE} \
--train_steps=${TRAINSTEPS} \
--warmup_steps=${WARMUPSTEPS} \
--adjust_loss=${ADJUSTLOSS} \
--rerank=${RERANK} \
--save_steps=${SAVESTEPS} \
--do_train=false \
--do_predict=true \
--do_export=false \
--overwrite_data=false 
```

Creating a condensed document through the minimum span coverage algorithm 
```
python3 answers_to_text.py
```

Second reading (document reader)

```
CUDA_VISIBLE_DEVICES=${GPUDEVICE} python electra_answer_as_text.py \
--vocab_file=${MODELDIR}/vocab.txt \
--model_config_path=${MODELDIR}/bert_config.json \
--init_checkpoint=${MODELDIR}/reanswer/model.ckpt-20000 \
--task_name=${TASKNAME} \
--random_seed=${RANDOMSEED} \
--predict_tag=reanswer \
--do_lower_case=${DOLOWERCASE} \
--data_dir=${DATADIR}/ \
--output_dir=${OUTPUTDIR}/data_reanswer \
--model_dir=${OUTPUTDIR}/checkpoint \
--export_dir=${OUTPUTDIR}/export \
--num_turn=${NUMTURN} \
--max_seq_length=${SEQLEN} \
--max_query_length=${QUERYLEN} \
--max_answer_length=${ANSWERLEN} \
--train_batch_size=${BATCHSIZE} \
--predict_batch_size=${BATCHSIZE} \
--num_hosts=1 \
--num_core_per_host=1 \
--learning_rate=${LEARNINGRATE} \
--train_steps=${TRAINSTEPS} \
--warmup_steps=${WARMUPSTEPS} \
--adjust_loss=${ADJUSTLOSS} \
--rerank=${RERANK} \
--save_steps=${SAVESTEPS} \
--do_train=false \
--do_predict=true \
--do_export=false \
--overwrite_data=false
```

Post-processing (answer aggregation and voting)
```
python tool/convert_quac_cross.py \
--input_file=./predict.best.detail.json \
--output_file=${OUTPUTDIR}/data/predict.${PREDICTTAG}.span.json \
--answer_threshold=${ANSWERTHRESHOLD}
```

The above steps are integrated to a shell script run.sh. you can directly run it through:
```
bash run.sh
```

## Results

The QuAC leaderboard is [here](https://quac.ai/). RoR ranks the 1st place on the QuAC leaderboard. Please refer to our paper to get more ablation results. 


