SOURCE_MODEL=trained/elman_shakespeare_2_1024_178000.t7
SOURCE_LOG=trained/elman_shakespeare_2_1024_model_log
MOD=$(SOURCE_LOG).mod
PY=$(SOURCE_LOG).py
INVERSE=$(SOURCE_LOG)_inverse.dat

all:  $(MOD) $(INVERSE)

$(SOURCE_LOG):  $(SOURCE_MODEL)
	cd torch-rnn && th ../PRINT_MODEL.lua -checkpoint ../$(SOURCE_MODEL) -gpu -1 > ../$@

$(PY):
$(MOD):  $(SOURCE_LOG) logToAMPL.py
	python logToAMPL.py $<

$(INVERSE): $(PY) invertWeightMatrices.py
	python invertWeightMatrices.py $(SOURCE_LOG)

