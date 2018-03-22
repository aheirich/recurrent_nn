#!/bin/bash
docker run --rm -v ${MACHINELEARNING}/crossMount/torch-rnn-docker:/root/torch-rnn/external -ti crisbal/torch-rnn:base bash
