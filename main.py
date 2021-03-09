import numpy as np
import torch
import torch.nn as nn
import time
import os
from tool.dataLoader import *
from torch.utils.data import DataLoader
from models.models import domain2Vector, channelAttention
from models.ResNet3D import ResNet50TA, ResNet50TAPhase
from tool.averageMeter import AverageMeter, getAverageMeter, lossesUpdate
import sys
from tool.losses import CrossEntropyLabelSmooth, TripletLoss, CenterLoss, negativeEntropy
from tool.evalMetrics import evaluate
from random import sample
from colorama import Style, Fore, Back
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, required=False, default="./data/cifar10", help="Data directory")
parser.add_argument('--batchSize', default=32, type=int, help='training batch size')
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--decayGamma', default=0.8, type=float)
parser.add_argument('--decayStepSize', default=20, type=int)
parser.add_argument('--mode', type=str, default="training")
parser.add_argument('--trainingCheckpoint', type=str, default="./checkpoint/model.pth")
parser.add_argument('--testingCheckpoint', type=str, default="./checkpoint/model.pth")
parser.add_argument('--dataPath', type=str, default="/data2/lab50_dataset/*")
parser.add_argument('--maxEpoch', default=150, type=int)
parser.add_argument('--testFreq', default=1, type=int)
parser.add_argument('--csiChannel', default=30, type=int)
parser.add_argument('--workers', default=0, type=int)
parser.add_argument('--clipLen', default=5, type=int)
parser.add_argument('--numAction', default=4, type=int)
parser.add_argument('--trainIdsRange', type=int, nargs='+')
parser.add_argument('--testIdsRange', type=int, nargs='+')

args = parser.parse_args()

useGpu = torch.cuda.is_available()


def saveCheckpoint(model, optimizer, **kwargs):
	if not os.path.exists('./checkpoint'): os.makedirs('./checkpoint')
	torch.save({
		'epoch': kwargs['epoch'] + 1,
		'state_dict': model.state_dict(),
		'optimizer': optimizer.state_dict(),
		'best': kwargs['best']
	}, args.trainingCheckpoint)


def loadCheckpoint(model, mode, optimizer=None):
	if mode == 'training': checkpointPath = args.trainingCheckpoint
	else: checkpointPath = args.testingCheckpoint

	load = True
	if os.path.isfile(checkpointPath):
		checkpoint = torch.load(checkpointPath)
		startEpoch = checkpoint['epoch']
		bestScore = checkpoint['best']
		model.load_state_dict(checkpoint['state_dict'])
		if optimizer: optimizer.load_state_dict(checkpoint['optimizer'])
		print(Fore.RED + "=> loaded checkpoint '{}' (epoch {}, bestScore {:.4f})"\
		.format(checkpointPath, checkpoint['epoch'], checkpoint['best']) + Style.RESET_ALL)
	else:
		bestScore = 0
		startEpoch = 0
		print(Fore.RED + "=> no checkpoint found at '{}'".format(checkpointPath) + Style.RESET_ALL)
		load = False

	return model, optimizer, startEpoch, bestScore, load


def trainDisentangleNoClassifier(model, trainingLoader, optimizer, optimizerCentloss, scheduler, epoch, **loss):
	start = time.time()
	print(Fore.CYAN + "==> Training")
	print("==> Epoch {}/{}".format(epoch, args.maxEpoch))
	print("==> Learning Rate = {}".format(optimizer.param_groups[0]['lr']))
	losses = getAverageMeter(numLosses=5)
	model.train()

	pBar = tqdm(trainingLoader, desc='Training')
	for batchIndex, (csiDataDataData1, personId1, actionId1, csiDataDataData2, personId2, actionId2) in enumerate(pBar):
		b, t, c, f, h, w = csiDataDataData1.shape
		if useGpu:
			csiDataDataData1 = csiDataDataData1.cuda()
			personId1 = personId1.cuda()
			actionId1 = actionId1.cuda()
			csiDataDataData2 = csiDataDataData2.cuda()
			personId2 = personId2.cuda()
			actionId2 = actionId2.cuda()

		stage1, stage2 = model(csiDataDataData1, csiDataDataData2)
		(x1Id, x1Action, x2Id, x2Action, x1IdFeature, x2IdFeature, x1IdEncoder, x2IdEncoder) = stage1
		(x1Id_, x1Action_, x2Id_, x2Action_, x1Id_x1Action, x2Id_x2Action, x1IdFeature_, x2IdFeature_, x1IdEncoder_, x2IdEncoder_) = stage2

		idLoss = loss['id'](x1Id, personId1) + loss['id'](x2Id, personId2) + \
					loss['id'](x1Id_, personId1) + loss['id'](x2Id_, personId2)
		triLoss = loss['tri'](x1IdFeature, personId1) + loss['tri'](x2IdFeature, personId2) + \
					loss['tri'](x1IdFeature_, personId1) + loss['tri'](x2IdFeature_, personId2)
		centLoss = loss['center'](x1IdFeature, personId1) + loss['center'](x2IdFeature, personId2) + \
					loss['center'](x1IdFeature_, personId1) + loss['center'](x2IdFeature_, personId2)
		actionLoss = loss['action'](x1Action, actionId1) + loss['action'](x2Action, actionId2) + \
					loss['action'](x1Action_, actionId1) + loss['action'](x2Action_, actionId2)
		cycleLoss = loss['l2'](x1Id_x1Action, csiDataDataData1.permute(1, 0, 2, 3, 4, 5).reshape(-1, c, f, h, w)) + \
					loss['l2'](x2Id_x2Action, csiDataDataData2.permute(1, 0, 2, 3, 4, 5).reshape(-1, c, f, h, w))

		totalLoss = idLoss + triLoss + centLoss/10 + actionLoss + cycleLoss
		losses = lossesUpdate(losses, args.batchSize, [idLoss, triLoss, centLoss/10, actionLoss, cycleLoss])


		pBar.set_postfix({'id': '{:.3f}'.format(losses[0].avg), 'tri': '{:.3f}'.format(losses[1].avg), 'cent': '{:.3f}'.format(losses[2].avg),\
		 'action': '{:.3f}'.format(losses[3].avg), 'cycle': '{:.3f}'.format(losses[4].avg)})

		optimizer.zero_grad()
		optimizerCentloss.zero_grad()
		totalLoss.backward()
		optimizer.step()
		optimizerCentloss.step()

	scheduler.step()

	endl = time.time()
	print('Costing time:', (endl-start)/60)
	t = time.localtime()
	current_time = time.strftime("%H:%M:%S", t)
	print('Current time:', current_time)
	print(Style.RESET_ALL, end='')
	print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)


def trainDisentangle(model, classifier, trainingLoader, optimizer, optimizerCentloss, scheduler, epoch, **loss):
	start = time.time()
	print(Fore.CYAN + "==> Training")
	print("==> Epoch {}/{}".format(epoch, args.maxEpoch))
	print("==> Learning Rate = {}".format(optimizer.param_groups[0]['lr']))
	losses = getAverageMeter(numLosses=6)
	model.train()
	classifier.eval()

	pBar = tqdm(trainingLoader, desc='Training')
	for batchIndex, (csiDataDataData1, personId1, actionId1, csiDataDataData2, personId2, actionId2) in enumerate(pBar):
		b, t, c, f, h, w = csiDataDataData1.shape
		if useGpu:
			csiDataDataData1 = csiDataDataData1.cuda()
			personId1 = personId1.cuda()
			actionId1 = actionId1.cuda()
			#csiDataDataData2 = csiDataDataData2.cuda()
			#personId2 = personId2.cuda()
			#actionId2 = actionId2.cuda()

		x1IdFeature, x1Id, x1Action = model(csiDataDataData1, csiDataDataData2)
		# (x1Id, x1Action, x2Id, x2Action, x1IdFeature, x2IdFeature, x1IdEncoder, x2IdEncoder) = stage1
		# (x1Id_, x1Action_, x2Id_, x2Action_, x1Id_x1Action, x2Id_x2Action, x1IdFeature_, x2IdFeature_, x1IdEncoder_, x2IdEncoder_) = stage2
		# print(personId1)
		# print(x1Id)
		exit()
		# _, envId1 = classifier(x1IdEncoder, b, t)
		# _, envId2 = classifier(x2IdEncoder, b, t)
		# _, envId1_ = classifier(x1IdEncoder_, b, t)
		# _, envId2_ = classifier(x2IdEncoder_, b, t)

		idLoss = loss['id'](x1Id, personId1) #+ \
					# loss['id'](x1Id_, personId1) + loss['id'](x2Id_, personId2)
		triLoss = loss['tri'](x1IdFeature, personId1) #+ \
		# 			loss['tri'](x1IdFeature_, personId1) + loss['tri'](x2IdFeature_, personId2)
		centLoss = loss['center'](x1IdFeature, personId1) #+ \
		# 			loss['center'](x1IdFeature_, personId1) + loss['center'](x2IdFeature_, personId2)
		actionLoss = loss['action'](x1Action, actionId1) #+ \
					# loss['action'](x1Action_, actionId1) + loss['action'](x2Action_, actionId2)
		# cycleLoss = loss['l2'](x1Id_x1Action, csiDataDataData1.permute(1, 0, 2, 3, 4, 5).reshape(-1, c, f, h, w)) + \
		# 			loss['l2'](x2Id_x2Action, csiDataDataData2.permute(1, 0, 2, 3, 4, 5).reshape(-1, c, f, h, w))
		# negEntLoss = loss['env'](envId1) + loss['env'](envId2) + loss['env'](envId1_) + loss['env'](envId2_)

		totalLoss = idLoss + triLoss + centLoss/10 + actionLoss # + cycleLoss + negEntLoss/20 + triLoss + centLoss/10
		losses = lossesUpdate(losses, args.batchSize, [idLoss, triLoss, centLoss/10, actionLoss]) #triLoss, centLoss/10, actionLoss, cycleLoss, negEntLoss/20])


		pBar.set_postfix({'id': '{:.3f}'.format(losses[0].avg), 'tri': '{:.3f}'.format(losses[1].avg), 'cent': '{:.3f}'.format(losses[2].avg),\
		 'action': '{:.3f}'.format(losses[3].avg)}) #, 'cycle': '{:.3f}'.format(losses[4].avg), 'env': '{:.3f}'.format(losses[5].avg)})

		optimizer.zero_grad()
		optimizerCentloss.zero_grad()
		totalLoss.backward()
		optimizer.step()
		optimizerCentloss.step()

	scheduler.step()

	endl = time.time()
	print('Costing time:', (endl-start)/60)
	t = time.localtime()
	current_time = time.strftime("%H:%M:%S", t)
	print('Current time:', current_time)
	print(Style.RESET_ALL, end='')
	print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)


def test(model, queryLoader, galleryLoader, ranks=[1, 3, 5, 7]):
	start = time.time()
	print(Fore.GREEN + "==> Testing")
	model.eval()
	queryFeat, queryPersonId = [], []
	for batchIndex, (csiDataData, pids, _) in enumerate(tqdm(queryLoader, desc='Extracted features for query set')):
		if useGpu: csiDataData = csiDataData.cuda()
		b, n, t, c, f, h, w = csiDataData.size()

		assert(b==1)
		csiDataData = csiDataData.view(b*n, t, c, f, h, w)

		outputs, features, _ = model(csiDataData)
		features = torch.mean(features, 0)
		features = features.data.cpu()

		outputs = torch.mean(outputs, 0)
		outputs = outputs.unsqueeze(0)

		queryFeat.append(features)
		queryPersonId.extend(pids)
	queryFeat = torch.stack(queryFeat)
	queryPersonId = np.asarray(queryPersonId)
	print("Extracted features for query set, obtained {}-by-{} matrix".format(queryFeat.size(0), queryFeat.size(1)))

	galleryFeat, galleryPersonId = [], []
	for batchIndex, (csiData, pids, _) in enumerate(tqdm(galleryLoader, desc='Extracted features for gallery set')):
		if useGpu: csiData = csiData.cuda()

		b, n, t, c, f, h, w = csiData.size()
		assert(b==1)
		csiData = csiData.view(b*n, t, c, f, h, w)

		outputs, features, _ = model(csiData)
		features = torch.mean(features, 0)
		features = features.data.cpu()

		outputs = torch.mean(outputs, 0)
		outputs = outputs.unsqueeze(0)

		galleryFeat.append(features)
		galleryPersonId.extend(pids)
	galleryFeat = torch.stack(galleryFeat)
	galleryPersonId = np.asarray(galleryPersonId)

	print("Extracted features for gallery set, obtained {}-by-{} matrix".format(galleryFeat.size(0), galleryFeat.size(1)))
	
	m, n = queryFeat.size(0), galleryFeat.size(0)  
	distanceMatrix = torch.pow(queryFeat, 2).sum(dim=1, keepdim=True).expand(m, n) + \
					torch.pow(galleryFeat, 2).sum(dim=1, keepdim=True).expand(n, m).t()
	distanceMatrix.addmm_(1, -2, queryFeat, galleryFeat.t())
	distanceMatrix = distanceMatrix.numpy()
	print("Computing distance matrix, obtained {}-by-{} matrix".format(distanceMatrix.shape[0], distanceMatrix.shape[1]))

	cmc, mAP = evaluate(distanceMatrix, queryPersonId, galleryPersonId)
	
	print("\nResults")
	print("------------------")
	print("mAP: {:.1%}".format(mAP))
	print("CMC curve")
	for r in ranks:
		print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
	print("------------------")
	print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)

	return cmc[0]


def mainDisentanglePhaseNoClassifier():
	if args.mode == 'training':
		trainingData = preprocessPhase(args.dataPath, args.clipLen, 'training', trainIds=range(args.numIds))
		trainingLoader = DataLoader(dataset = DatasetPhase(trainingData, 'training', testIdStart=0), batch_size = args.batchSize, 
									num_workers = args.workers, drop_last = True, shuffle = True)

		testIdStart = 0 if args.numIds == 50 else args.numIds
		queryData, galleryData = preprocessPhase(args.dataPath, args.clipLen, 'testing', testIds=range(testIdStart, 50))
		queryLoader = DataLoader(dataset = DatasetPhase(queryData, 'testing', testIdStart=testIdStart), batch_size = 1, num_workers = 0,
								drop_last = True, shuffle = True)
		galleryLoader = DataLoader(dataset = DatasetPhase(galleryData, 'testing', testIdStart=testIdStart), batch_size = 1, num_workers = 0,
								drop_last = True, shuffle = True)

		model = disentanglePhase(args.csiChannel, args.numIds, featDim=32)
		model.weights_init()
		mseLoss = nn.MSELoss()
		idEntropyLoss = CrossEntropyLabelSmooth(args.numIds, use_gpu=useGpu)
		actionEntropyLoss = CrossEntropyLabelSmooth(args.numAction, use_gpu=useGpu)
		tripletLoss = TripletLoss(args.numIds)
		centerLoss = CenterLoss(args.numIds, 32*6, useGpu)
		optimizerCentloss = torch.optim.SGD(centerLoss.parameters(), lr=0.05)
		
		if useGpu:
			model = model.cuda()
			mseLoss = mseLoss.cuda()
			idEntropyLoss = idEntropyLoss.cuda()
			actionEntropyLoss = actionEntropyLoss.cuda()
			tripletLoss = tripletLoss.cuda()
			centerLoss = centerLoss.cuda()

		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decayStepSize, gamma=args.decayGamma)

		model, optimizer, startEpoch, bestScore, _ = loadCheckpoint(model, args.mode, optimizer)
		print(Fore.RED + '=> Training Ids\t: {} ~ {}'.format(0, args.numIds-1))
		print('=> Testing Ids\t: {} ~ {}'.format(testIdStart, 49) + Style.RESET_ALL)
		for epoch in range(startEpoch, args.maxEpoch):
			start = time.time()
			trainDisentangleNoClassifier(model, trainingLoader, optimizer, optimizerCentloss, scheduler, epoch,
				l2=mseLoss, id=idEntropyLoss, action=actionEntropyLoss, tri=tripletLoss, center=centerLoss)
			if epoch % args.testFreq == 0:
				rank1 = test(model, queryLoader, galleryLoader)
				if rank1 > bestScore:
					if not os.path.exists('./checkpoint'): os.makedirs('./checkpoint')
					print(Fore.RED + 'Rank1: {:.3f}  >=  Best Score {:.3f}'.format(rank1, bestScore))
					print('Update model!!!')
					print("---------------------------------------------------------------" + Style.RESET_ALL)
					bestScore = rank1
					saveCheckpoint(model, optimizer, epoch=epoch, best=rank1)


	elif args.mode == 'testing':
		testIdStart = 0 if args.numIds == 50 else args.numIds
		queryData, galleryData = preprocessPhase(args.dataPath, args.clipLen, 'testing', testIds=range(testIdStart, 50))
		queryLoader = DataLoader(dataset = DatasetPhase(queryData, 'testing', testIdStart=testIdStart), batch_size = 1, num_workers = 0,
								drop_last = True, shuffle = True)
		galleryLoader = DataLoader(dataset = DatasetPhase(galleryData, 'testing', testIdStart=testIdStart), batch_size = 1, num_workers = 0,
								drop_last = True, shuffle = True)

		# load the model
		model = disentanglePhase(args.csiChannel, args.numIds, featDim=32)
		model, optimizer, startEpoch, bestScore, load = loadCheckpoint(model, args.mode)
		if not load: exit()

		if useGpu: model = model.cuda()
		test(model, queryLoader, galleryLoader)

	else:
		print('Error: mode should be training or testing')
		return 0


def main():
	if args.mode == 'training':
		trainIdsRange = [i-1 for i in range(args.trainIdsRange[0], args.trainIdsRange[1]+1)]
		trainingData = preprocessPhase(args.dataPath, args.clipLen, 'training', idsRange=trainIdsRange)
		trainingLoader = DataLoader(dataset = DatasetPhase(trainingData, 'training', idStart=args.trainIdsRange[0]), 
									batch_size = args.batchSize, num_workers = args.workers, drop_last = True, shuffle = True)

		testIdsRange = [i-1 for i in range(args.testIdsRange[0], args.testIdsRange[1]+1)]
		queryData, galleryData = preprocessPhase(args.dataPath, args.clipLen, 'testing', idsRange=testIdsRange)
		queryLoader = DataLoader(dataset = DatasetPhase(queryData, 'testing', idStart=args.testIdsRange[0]), batch_size = 1, num_workers = 0,
								drop_last = True, shuffle = True)
		galleryLoader = DataLoader(dataset = DatasetPhase(galleryData, 'testing', idStart=args.testIdsRange[0]), batch_size = 1, num_workers = 0,
								drop_last = True, shuffle = True)

		model = domain2Vector(args.csiChannel, len(trainIdsRange), featDim=32)
		model.weights_init()
		mseLoss = nn.MSELoss()
		idEntropyLoss = CrossEntropyLabelSmooth(len(trainIdsRange), use_gpu=useGpu)
		actionEntropyLoss = CrossEntropyLabelSmooth(args.numAction, use_gpu=useGpu)
		tripletLoss = TripletLoss(len(trainIdsRange))
		centerLoss = CenterLoss(len(trainIdsRange), 2048, useGpu)
		optimizerCentloss = torch.optim.SGD(centerLoss.parameters(), lr=0.05)
		negEntropyLoss = negativeEntropy(useGpu)
		classifier = channelAttention(featDim=2048)
		
		if useGpu:
			model = model.cuda()
			mseLoss = mseLoss.cuda()
			idEntropyLoss = idEntropyLoss.cuda()
			actionEntropyLoss = actionEntropyLoss.cuda()
			tripletLoss = tripletLoss.cuda()
			centerLoss = centerLoss.cuda()
			negEntropyLoss = negEntropyLoss.cuda()
			classifier = classifier.cuda()

		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decayStepSize, gamma=args.decayGamma)

		# load the model
		# checkpoint_ = torch.load('/home/culiver/PycharmProjects/WiSPPN_2/weights_AAAI/Deasfn_noise_discriminator_phase_12_05_1.pkl')
		# classifier.load_state_dict(checkpoint_['state_dict'])
		# for k,v in classifier.named_parameters():
		# 	v.requires_grad=False

		model, optimizer, startEpoch, bestScore, _ = loadCheckpoint(model, args.mode, optimizer)
		print(Fore.RED + '=> Training Ids\t: {} ~ {}'.format(args.trainIdsRange[0], args.trainIdsRange[1]))
		print('=> Testing Ids\t: {} ~ {}'.format(args.testIdsRange[0], args.testIdsRange[1]) + Style.RESET_ALL)




		for epoch in range(startEpoch, args.maxEpoch):
			start = time.time()
			trainDisentangle(model, classifier, trainingLoader, optimizer, optimizerCentloss, scheduler, epoch,
				l2=mseLoss, id=idEntropyLoss, action=actionEntropyLoss, tri=tripletLoss, center=centerLoss, env=negEntropyLoss)
			if epoch % args.testFreq == 0:
				rank1 = test(model, queryLoader, galleryLoader)
				if rank1 > bestScore:
					if not os.path.exists('./checkpoint'): os.makedirs('./checkpoint')
					print(Fore.RED + 'Rank1: {:.3f}  >=  Best Score {:.3f}'.format(rank1, bestScore))
					print('Update model!!!')
					print("---------------------------------------------------------------" + Style.RESET_ALL)
					bestScore = rank1
					saveCheckpoint(model, optimizer, epoch=epoch, best=rank1)


	elif args.mode == 'testing':
		testIdsRange = [i-1 for i in range(args.testIdsRange[0], args.testIdsRange[1]+1)]
		queryData, galleryData = preprocessPhase(args.dataPath, args.clipLen, 'testing', idsRange=testIdsRange)
		queryLoader = DataLoader(dataset = DatasetPhase(queryData, 'testing', idStart=args.testIdsRange[0]), batch_size = 1, num_workers = 0,
								drop_last = True, shuffle = True)
		galleryLoader = DataLoader(dataset = DatasetPhase(galleryData, 'testing', idStart=args.testIdsRange[0]), batch_size = 1, num_workers = 0,
								drop_last = True, shuffle = True)

		# load the model
		model = disentangleNew(args.csiChannel, len(testIdsRange), featDim=32)
		model, optimizer, startEpoch, bestScore, load = loadCheckpoint(model, args.mode)
		if not load: exit()
		exit()

		if useGpu: model = model.cuda()
		test(model, queryLoader, galleryLoader)

	else:
		print('Error: mode should be training or testing')
		return 0


if __name__ == '__main__':
	# mainDisentanglePhaseNoClassifier()
	main()

