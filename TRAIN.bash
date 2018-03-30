#!/bin/bash
cd ${TORCH_RNN_PATH}
source setup.bash

if [[ -d /opt/torch-rnn ]] ; then cd /opt/torch-rnn ; fi
pwd

for NUM_LAYERS in 1 2 3 4 
do
  for NUM_UNITS_PER_LAYER in 128 256 512 
  do
    DESCRIPTOR="_${NUM_LAYERS}_${NUM_UNITS_PER_LAYER}"
    PREFIX="${TORCH_RNN_PATH}/cv"
    MODELNAME="elman_shakespeare"
    LOGFILE=${PREFIX}/${MODELNAME}.${DESCRIPTOR}.log
    CHECKPOINT_NAME=${PREFIX}/${MODELNAME}.${DESCRIPTOR}
    echo ===
    echo === case ${DESCRIPTOR} LOGFILE=${LOGFILE}
    echo ===
    th ${SINGULARITY_TORCH_RNN_PATH}/train.lua \
    -input_h5 ${TORCH_RNN_PATH}/data/tiny-shakespeare.h5 \
    -input_json ${TORCH_RNN_PATH}/data/tiny-shakespeare.json \
    -max_epochs 30 \
    -model_type rnn \
    -rnn_size ${NUM_UNITS_PER_LAYER} \
    -num_layers ${NUM_LAYERS} \
    -checkpoint_name ${CHECKPOINT_NAME} \
    -gpu -1 \
    | tee ${LOGFILE}
    echo ===
    echo === extract AMPL model and data to ${CHECKPOINT_NAME}
    echo ===
    CK_TMP_FILE=${TORCH_RNN_PATH}/.ck_tmp_file
    rm -f ${CK_TMP_FILE}
    ls -1 ${CHECKPOINT_NAME}_*.t7 | sed -e "s:${CHECKPOINT_NAME}_::" | sed -e "s:.t7::" > ${CK_TMP_FILE}
    BATCH_NUMBER=`sort -n < ${CK_TMP_FILE} | tail -1`
    CHECKPOINT_PATH="${CHECKPOINT_NAME}_${BATCH_NUMBER}.t7"
    TMP_LOG=${TORCH_RNN_PATH}/.tmplog
    th ${TORCH_RNN_PATH}/PRINT_MODEL.lua -checkpoint ${CHECKPOINT_PATH} > ${TMP_LOG}
    python ${TORCH_RNN_PATH}/logToAMPL.py ${TMP_LOG} ${CHECKPOINT_NAME}
  done
done
