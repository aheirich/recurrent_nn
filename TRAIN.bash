th train.lua \
-input_h5 data/tiny-shakespeare.h5 \
-input_json data/tiny-shakespeare.json \
-max_epochs 100 \
-model_type rnn \
-checkpoint_name cv/elman_shakespeare \
-gpu -1
