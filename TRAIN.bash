#!/bin/bash
cd ${TORCH_RNN_PATH}
source setup.bash

LEARNING_RATE=0.0001
if [[ "$1" != "" ]]
then
  LEARNING_RATE=$1
fi

MODELNAME="elman_shakespeare"

if [[ "$2" != "" ]]
then
  MODELNAME=$2
fi


if [[ -d /opt/torch-rnn ]] ; then cd /opt/torch-rnn ; fi
pwd

mkdir -p ${SCRATCH}/cv

for NUM_LAYERS in 2
do
  for NUM_UNITS_PER_LAYER in 128 
  do
    DESCRIPTOR="_${NUM_LAYERS}_${NUM_UNITS_PER_LAYER}"
    PREFIX="${SCRATCH}/cv"
    LOGFILE=${PREFIX}/${MODELNAME}.${DESCRIPTOR}.log
    CHECKPOINT_NAME=${PREFIX}/${MODELNAME}.${DESCRIPTOR}

    echo ===
    echo === case ${DESCRIPTOR} LOGFILE=${LOGFILE}
    echo ===

    th ${SINGULARITY_TORCH_RNN_PATH}/train.lua \
    -input_h5 ${TORCH_RNN_PATH}/data/tiny-shakespeare.h5 \
    -input_json ${TORCH_RNN_PATH}/data/tiny-shakespeare.json \
    -max_epochs 500 \
    -model_type rnn \
    -rnn_size ${NUM_UNITS_PER_LAYER} \
    -num_layers ${NUM_LAYERS} \
    -checkpoint_name ${CHECKPOINT_NAME} \
    -learning_rate ${LEARNING_RATE} \
    -lr_decay_every 50 \
    -lr_decay_factor 0.6 \
    | tee ${LOGFILE}

    echo ===
    echo === extract AMPL model and data to ${CHECKPOINT_NAME}
    echo ===

    CK_TMP_FILE=${TORCH_RNN_PATH}/.ck_tmp_file
    rm -f ${CK_TMP_FILE}
    ls -1 ${CHECKPOINT_NAME}_*.t7 | sed -e "s:${CHECKPOINT_NAME}_::" | sed -e "s:.t7::" > ${CK_TMP_FILE}
    BATCH_NUMBER=`sort -n < ${CK_TMP_FILE} | tail -1`
    CHECKPOINT_PATH="${CHECKPOINT_NAME}_${BATCH_NUMBER}.t7"
    TMP_LOG=${TORCH_RNN_PATH}/${MODELNAME}.${DESCRIPTOR}_model.log
    th ${TORCH_RNN_PATH}/PRINT_MODEL.lua -checkpoint ${CHECKPOINT_PATH} > ${TMP_LOG}
    python ${TORCH_RNN_PATH}/logToAMPL.py ${TMP_LOG} ${CHECKPOINT_NAME}

  done
done

