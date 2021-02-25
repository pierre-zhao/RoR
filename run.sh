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
echo "gpu device        = ${GPUDEVICE}"
echo "num gpus          = ${NUMGPUS}"
echo "task name         = ${TASKNAME}"
echo "random seed       = ${RANDOMSEED}"
echo "predict tag       = ${PREDICTTAG}"
echo "model dir         = ${MODELDIR}"
echo "data dir          = ${DATADIR}"
echo "output dir        = ${OUTPUTDIR}"
echo "num turn          = ${NUMTURN}"
echo "seq len           = ${SEQLEN}"
echo "query len         = ${QUERYLEN}"
echo "answer len        = ${ANSWERLEN}"
echo "batch size        = ${BATCHSIZE}"
echo "learning rate     = ${LEARNINGRATE}"
echo "train steps       = ${TRAINSTEPS}"
echo "warmup steps      = ${WARMUPSTEPS}"
echo "save steps        = ${SAVESTEPS}"
echo "answer threshold  = ${ANSWERTHRESHOLD}"
echo "middle loss       = ${MIDDLELOSS}"
echo "layer decay       = ${LAYERDECAY}"
echo "decay method      = ${DECAYMETHOD}"
echo "adjust loss       = ${ADJUSTLOSS}"
echo "label smoothing   = ${LABELSMOOTHING}"
alias python=python3


# CUDA_VISIBLE_DEVICES=${GPUDEVICE} python electra_quac.py \
# --vocab_file=${MODELDIR}/vocab.txt \
# --model_config_path=${MODELDIR}/bert_config.json \
# --init_checkpoint=${MODELDIR}/model.ckpt-20000 \
# --task_name=${TASKNAME} \
# --random_seed=${RANDOMSEED} \
# --predict_tag=${PREDICTTAG} \
# --do_lower_case=${DOLOWERCASE} \
# --data_dir=${DATADIR}/ \
# --output_dir=${OUTPUTDIR}/data \
# --model_dir=${OUTPUTDIR}/checkpoint  \
# --export_dir=${OUTPUTDIR}/export \
# --num_turn=${NUMTURN} \
# --max_seq_length=${SEQLEN} \
# --max_query_length=${QUERYLEN} \
# --max_answer_length=${ANSWERLEN} \
# --train_batch_size=${BATCHSIZE} \
# --predict_batch_size=${BATCHSIZE} \
# --num_hosts=1 \
# --num_core_per_host=1 \
# --learning_rate=${LEARNINGRATE} \
# --train_steps=${TRAINSTEPS} \
# --warmup_steps=${WARMUPSTEPS} \
# --adjust_loss=${ADJUSTLOSS} \
# --rerank=${RERANK} \
# --save_steps=${SAVESTEPS} \
# --do_train=false \
# --do_predict=true \
# --do_export=false \
# --overwrite_data=false 


# python answers_to_text.py

# PREDICTTAG=reanswer
# CUDA_VISIBLE_DEVICES=${GPUDEVICE} python electra_answer_as_text.py \
# --vocab_file=${MODELDIR}/vocab.txt \
# --model_config_path=${MODELDIR}/bert_config.json \
# --init_checkpoint=${MODELDIR}/reanswer/model.ckpt-20000 \
# --task_name=${TASKNAME} \
# --random_seed=${RANDOMSEED} \
# --predict_tag=${PREDICTTAG} \
# --do_lower_case=${DOLOWERCASE} \
# --data_dir=${DATADIR}/ \
# --output_dir=${OUTPUTDIR}/data_reanswer \
# --model_dir=${OUTPUTDIR}/checkpoint \
# --export_dir=${OUTPUTDIR}/export \
# --num_turn=${NUMTURN} \
# --max_seq_length=${SEQLEN} \
# --max_query_length=${QUERYLEN} \
# --max_answer_length=${ANSWERLEN} \
# --train_batch_size=${BATCHSIZE} \
# --predict_batch_size=${BATCHSIZE} \
# --num_hosts=1 \
# --num_core_per_host=1 \
# --learning_rate=${LEARNINGRATE} \
# --train_steps=${TRAINSTEPS} \
# --warmup_steps=${WARMUPSTEPS} \
# --adjust_loss=${ADJUSTLOSS} \
# --rerank=${RERANK} \
# --save_steps=${SAVESTEPS} \
# --do_train=false \
# --do_predict=true \
# --do_export=false \
# --overwrite_data=false


PREDICTTAG=best
# python tool/convert_quac_cross.py \
# --input_file=${OUTPUTDIR}/data/predict.${PREDICTTAG}.detail.json \
# --output_file=${OUTPUTDIR}/data/predict.${PREDICTTAG}.span.json \
# --answer_threshold=${ANSWERTHRESHOLD}
python tool/convert_quac_cross.py \
--input_file=./predict.best.detail.json \
--output_file=${OUTPUTDIR}/data/predict.${PREDICTTAG}.span.json \
--answer_threshold=${ANSWERTHRESHOLD}
###
###
# python tool/convert_quac_rank.py \
# --input_file=${OUTPUTDIR}/data/predict.${PREDICTTAG}.detail.json \
# --output_file=${OUTPUTDIR}/data/predict.${PREDICTTAG}.span.json \
# --answer_threshold=${ANSWERTHRESHOLD}
#
rm ${OUTPUTDIR}/data/predict.${PREDICTTAG}.eval.json

python tool/eval_quac.py \
--val_file=${DATADIR}/dev-${TASKNAME}.json \
--model_output=${OUTPUTDIR}/data/predict.${PREDICTTAG}.span.json \
--o ${OUTPUTDIR}/data/predict.${PREDICTTAG}.eval.json