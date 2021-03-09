import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
import glob
import csv
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import hdf5storage
from random import randint


class Dataset(Dataset):

	def __init__(self, allClip, mode):
		self.allClip = allClip
		self.action = {'hand': 0, 'walk': 1, 'jump': 2, 'run': 3}
		self.mode = mode

	def __len__(self):
		return len(self.allClip)

	def __getitem__(self, index):
		if self.mode == 'training':
			csiData1 = []
			for frame in self.allClip[index]:
				data = hdf5storage.loadmat(frame, variable_names={'csi_serial'})
				csiData1.append(torch.from_numpy(data['csi_serial']).type(torch.FloatTensor).permute(1, 0, 2, 3))
			csiData1 = torch.stack(csiData1)
			personId1 = int(self.allClip[index][0].split("/")[-4])-1
			actionId1 = self.action[self.allClip[index][0].split("/")[-3]]

			csiData2 = []
			idx = randint(0, len(self.allClip)-1)
			for frame in self.allClip[idx]:
				data = hdf5storage.loadmat(frame, variable_names={'csi_serial'})
				csiData2.append(torch.from_numpy(data['csi_serial']).type(torch.FloatTensor).permute(1, 0, 2, 3))
			csiData2 = torch.stack(csiData2)
			personId2 = int(self.allClip[idx][0].split("/")[-4])-1
			actionId2 = self.action[self.allClip[idx][0].split("/")[-3]]

			return csiData1, personId1, actionId1, csiData2, personId2, actionId2

		elif self.mode == 'testing':
			clips = []
			for clip in self.allClip[index]:
				csiData = []
				for frame in clip:
					data = hdf5storage.loadmat(frame, variable_names={'csi_serial'})
					csiData.append(torch.from_numpy(data['csi_serial']).type(torch.FloatTensor).permute(1, 0, 2, 3))
				csiData = torch.stack(csiData)
				clips.append(csiData)
			clips = torch.stack(clips)

			personId = int(self.allClip[index][0][0].split("/")[-4])-1
			actionId = self.action[self.allClip[index][0][0].split("/")[-3]]

			return clips, personId, actionId

		else:
			print('Error: mode should be training or testing')


class DatasetPhase(Dataset):

	def __init__(self, allClip, mode, idStart):
		self.allClip = allClip
		self.action = {'hand': 0, 'walk': 1, 'jump': 2, 'run': 3}
		self.mode = mode
		self.idStart = idStart

	def __len__(self):
		return len(self.allClip)

	def __getitem__(self, index):
		if self.mode == 'training':
			csiData1 = []
			print(self.allClip[index])
			for frame in self.allClip[index]:
				data = hdf5storage.loadmat(frame, variable_names={'csi_serial_phase'})
				csiData1.append(torch.from_numpy(data['csi_serial_phase']).type(torch.FloatTensor).permute(1, 0, 2, 3))
			# print(csiData1)
			csiData1 = torch.stack(csiData1)
			# print(csiData1)

			personId1 = int(self.allClip[index][0].split("/")[-4]) - self.idStart
			actionId1 = self.action[self.allClip[index][0].split("/")[-3]]
			# print(personId1)
			csiData2 = []
			idx = randint(0, len(self.allClip)-1)
			for frame in self.allClip[idx]:
				data = hdf5storage.loadmat(frame, variable_names={'csi_serial_phase'})
				csiData2.append(torch.from_numpy(data['csi_serial_phase']).type(torch.FloatTensor).permute(1, 0, 2, 3))
			csiData2 = torch.stack(csiData2)
			personId2 = int(self.allClip[idx][0].split("/")[-4]) - self.idStart
			actionId2 = self.action[self.allClip[idx][0].split("/")[-3]]

			return csiData1, personId1, actionId1, csiData2, personId2, actionId2

		elif self.mode == 'testing':
			clips = []
			for clip in self.allClip[index]:
				csiData = []
				for frame in clip:
					data = hdf5storage.loadmat(frame, variable_names={'csi_serial_phase'})
					csiData.append(torch.from_numpy(data['csi_serial_phase']).type(torch.FloatTensor).permute(1, 0, 2, 3))
				csiData = torch.stack(csiData)
				clips.append(csiData)
			clips = torch.stack(clips)

			personId = int(self.allClip[index][0][0].split("/")[-4]) - self.idStart
			actionId = self.action[self.allClip[index][0][0].split("/")[-3]]

			return clips, personId, actionId

		else:
			print('Error: mode should be training or testing')


def preprocess(dataPath, clipLen, mode, testClipLen=4):
	folder = glob.glob(dataPath)
	folder.sort(key=takeFolderIndex)
	actions = ['hand', 'jump', 'run', 'walk']
	cnt = 0
	if mode == 'training':
		clip = []
		for folderIndex in range(50):
			for action in actions:
				filenames = glob.glob('{}/{}/train/*.mat'.format(folder[folderIndex], action))
				filenames.sort(key=takeCsiIndex)
				cnt += len(filenames)
				filenames = filenames[round(len(filenames)*0.1): round(len(filenames)*0.9)] 
				for i in range(0, int(len(filenames)/clipLen), 4):
					clip.append(filenames[i*clipLen: (i+1)*clipLen])
		return clip

	elif mode == 'testing':
		queryClip, galleryClip = [], []
		for folderIndex in range(50):
			for action in actions:
				filenames = glob.glob('{}/{}/test/*.mat'.format(folder[folderIndex], action))
				filenames.sort(key=takeCsiIndex)
				queryFilenames = filenames[round(len(filenames)*0.1): round(len(filenames)*0.9)] 
				for i in range(0, int(len(queryFilenames)/clipLen/testClipLen), 20):
					clip = []
					for j in range(testClipLen):
						clip.append(queryFilenames[i*clipLen*testClipLen + j*clipLen: i*clipLen*testClipLen + (j+1)*clipLen])
					queryClip.append(clip)

				filenames = glob.glob('{}/{}/train/*.mat'.format(folder[folderIndex], action))
				filenames.sort(key=takeCsiIndex)
				galleryFilenames = filenames[round(len(filenames)*0.1): round(len(filenames)*0.9)] 
				for i in range(0, int(len(galleryFilenames)/clipLen/testClipLen), 6):
					clip = []
					for j in range(testClipLen):
						clip.append(galleryFilenames[i*clipLen*testClipLen + j*clipLen: i*clipLen*testClipLen + (j+1)*clipLen])
					galleryClip.append(clip)

		return queryClip, galleryClip

	else:
		print('Error: mode should be training or testing')


def preprocessPhase(dataPath, clipLen, mode, idsRange, testClipLen=4):
	folder = glob.glob(dataPath)
	folder.sort(key=takeFolderIndex)
	actions = ['hand', 'jump', 'run', 'walk']

	if mode == 'training':
		clip = []
		for folderIndex in idsRange:
			for action in actions:
				filenames = glob.glob('{}/{}/train_phase/*.mat'.format(folder[folderIndex], action))
				filenames.sort(key=takeCsiIndex)
				filenames = filenames[round(len(filenames)*0.1): round(len(filenames)*0.9)] 
				for i in range(0, int(len(filenames)/clipLen), 4):
					clip.append(filenames[i*clipLen: (i+1)*clipLen])
		return clip

	elif mode == 'testing':
		queryClip, galleryClip = [], []
		for folderIndex in idsRange:
			for action in actions:
				filenames = glob.glob('{}/{}/test_phase/*.mat'.format(folder[folderIndex], action))
				filenames.sort(key=takeCsiIndex)
				queryFilenames = filenames[round(len(filenames)*0.1): round(len(filenames)*0.9)] 
				for i in range(0, int(len(queryFilenames)/clipLen/testClipLen), 20):
					clip = []
					for j in range(testClipLen):
						clip.append(queryFilenames[i*clipLen*testClipLen + j*clipLen: i*clipLen*testClipLen + (j+1)*clipLen])
					queryClip.append(clip)

				filenames = glob.glob('{}/{}/train_phase/*.mat'.format(folder[folderIndex], action))
				filenames.sort(key=takeCsiIndex)
				galleryFilenames = filenames[round(len(filenames)*0.1): round(len(filenames)*0.9)] 
				for i in range(0, int(len(galleryFilenames)/clipLen/testClipLen), 6):
					clip = []
					for j in range(testClipLen):
						clip.append(galleryFilenames[i*clipLen*testClipLen + j*clipLen: i*clipLen*testClipLen + (j+1)*clipLen])
					galleryClip.append(clip)

		return queryClip, galleryClip

	else:
		print('Error: mode should be training or testing')


def takeFolderIndex(elem):
	return int(elem.split("/")[3])


def takeCsiIndex(elem):
	return int(elem.split("/")[-1].split(".")[0])


