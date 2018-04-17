SOURCE_MODEL=trained/elman_shakespeare._2_1024_178000.t7
SOURCE_LOG=trained/elman_shakespeare._2_1024_model.log

all:  model_ampl.py model_inverse.dat

$(SOURCE_LOG):  $(SOURCE_MODEL)
	th PRINT_MODEL.lua -checkpoint $(SOURCE_MODEL) > $@

model_ampl.py:  $(SOURCE_LOG)
	python logToAmpl.py $<

model_inverse.dat: model_ampl.py invertWeightMatrices.py
	python invertWeightMatrices.py 

