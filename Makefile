trained/model_ampl.dat:	logToAMPL.py trained/PRINT_MODEL.log
	cat trained/PRINT_MODEL.log | python logToAMPL.py > trained/model_ampl.dat
