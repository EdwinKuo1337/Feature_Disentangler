trainArgs = dotdict({
	'LoadModel': True,
	'checkpoint': './0610113_2/checkpoint/popularity.pth',
	'TrainLabelFileName': './train2019.csv',
	'cudaDevice': 1,
	'workers': 4,
	'MaxEpoch': 150,
	'lr': 0.0001,
	'PrintFreq': 200,
	'BatchSize': 32,
	'NumClasses': 2,
	'DecayStepSize': 10,
	'DecayGamma': 0.8,
	'EvalEpoch': 3
})