import torch.nn as nn
import torch.nn.functional as F
import torch


# __all__ = ['disentangle', 'classifier']


def conv3x3x3(in_channels, out_channels, stride=1):
	return nn.Conv3d(in_channels, out_channels, kernel_size=3,
					 stride=stride, padding=1, bias=False)


def conv1x3x3(in_channels, out_channels, stride=1):
	return nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3),
					 stride=stride, padding=(0, 1, 1), bias=False)


def conv1x3x2(in_channels, out_channels, stride=1):
	return nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 2),
					 stride=stride, padding=(0, 1, 1), bias=False)


def ConvBlock(in_channels, out_channels, k, s):
	return nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=1, bias=False),
		nn.BatchNorm2d(out_channels),
		nn.ReLU(inplace=True)
		)


class ResidualBlock3D(nn.Module):
	expansion = 1

	def __init__(self, in_channels, out_channels, stride=1, downsample=None):
		super(ResidualBlock3D, self).__init__()
		self.conv1 = conv1x3x3(in_channels, out_channels, stride)
		self.bn1 = nn.BatchNorm3d(out_channels)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv1x3x3(out_channels, out_channels)
		self.bn2 = nn.BatchNorm3d(out_channels)
		self.downsample = downsample

	def forward(self, x):
		residual = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)

		out = self.bn2(out)
		if self.downsample:
			residual = self.downsample(x)
		out += residual
		out = self.relu(out)
		return out


class ResidualBlock3DPhase(nn.Module):
	expansion = 1

	def __init__(self, in_channels, out_channels, stride=1, downsample=None):
		super(ResidualBlock3DPhase, self).__init__()
		self.conv1 = conv1x3x2(in_channels, out_channels, stride)
		self.bn1 = nn.BatchNorm3d(out_channels)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv1x3x2(out_channels, out_channels)
		self.bn2 = nn.BatchNorm3d(out_channels)
		self.downsample = downsample

	def forward(self, x):
		residual = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)

		out = self.bn2(out)
		if self.downsample:
			residual = self.downsample(x)
		out += residual
		out = self.relu(out)
		return out


class encoder(nn.Module):
	def __init__(self, csiChannel, featDim):
		super(encoder, self).__init__()
		self.in_channels = csiChannel
		self.freqEncoder = nn.ModuleList()
		for i in range(1, 6):
			self.freqEncoder.append(
				nn.Sequential(
					nn.ConstantPad3d((0, 0, 0, 0, 5*i-1, 0), 0),
					nn.Conv3d(csiChannel, 64, kernel_size=(5 * i, 1, 1), padding=0),
					nn.BatchNorm3d(64),
					nn.ReLU(inplace=True),
					nn.Conv3d(64,featDim, kernel_size=(25, 1, 1)),
					nn.BatchNorm3d(featDim),
					nn.ReLU(inplace=True)
				)
			)

		self.spatialEncoder = nn.Sequential(
			self.make_layer(ResidualBlock3D, featDim, 4),
			nn.AvgPool3d((25, 1, 1))
		)

	def make_layer(self, block, out_channels, blocks, stride=1):
		downsample = None
		if (stride != 1) or (self.in_channels != out_channels * block.expansion):
			downsample = nn.Sequential(
				conv3x3x3(self.in_channels, out_channels * block.expansion, stride=stride),
				nn.BatchNorm3d(out_channels * block.expansion))
		layers = []
		layers.append(block(self.in_channels, out_channels, stride, downsample))
		self.in_channels = out_channels * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.in_channels, out_channels))
		return nn.Sequential(*layers)

	def forward(self, x):
		# Spatial Encoder
		SpatialFeat = self.spatialEncoder(x).squeeze(2)
		
		# Frequency Encoder
		FreqFeat = []
		for i in range(len(self.freqEncoder)):
			FreqFeat.append(self.freqEncoder[i](x).squeeze(2))
		FreqFeat = torch.cat(FreqFeat, dim=1)


		# Concatenate all the feature
		dualFeat = torch.cat([SpatialFeat, FreqFeat], dim=1)
		
		return dualFeat


class discriminator(nn.Module):
	def __init__(self, numClasses, featDim):
		super(discriminator, self).__init__()
		self.feat_dim = featDim # feature dimension
		self.middle_dim = 64 # middle layer dimension
		self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [3,3])
		self.attention_tconv = nn.Conv1d(self.middle_dim, 1, 3, padding=1)
		self.fc = nn.Linear(in_features=self.feat_dim, out_features=numClasses, bias=False)
		self.bottleneck = nn.BatchNorm1d(self.feat_dim)
		self.bottleneck.bias.requires_grad_(False)

	def forward(self, x, b, t):
		_, _, h, w = x.shape
		x = x.reshape(t, b, self.feat_dim, h, w)
		x = x.permute(1, 0, 2, 3, 4).reshape(-1, self.feat_dim, h, w)

		a = F.relu(self.attention_conv(x))
		a = a.reshape(b, t, self.middle_dim)
		a = a.permute(0,2,1)
		a = F.relu(self.attention_tconv(a))
		a = a.reshape(b, t)
		
		x = F.avg_pool2d(x, x.size()[2:])
		
		a = F.softmax(a, dim=1)
		x = x.reshape(b, t, -1)
		a = torch.unsqueeze(a, -1)
		a = a.expand(b, t, self.feat_dim)
		att_x = torch.mul(x, a)
		att_x = torch.sum(att_x, 1)

		att_x = self.bottleneck(att_x)
		out = self.fc(att_x)
		
		return att_x, out


class discriminatorPhase(nn.Module):
	def __init__(self, numClasses, featDim):
		super(discriminatorPhase, self).__init__()
		self.feat_dim = featDim # feature dimension
		self.middle_dim = 128 # middle layer dimension
		self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [3,2])
		self.attention_tconv = nn.Conv1d(self.middle_dim, 1, 3, padding=1)
		self.fc = nn.Linear(in_features=self.feat_dim, out_features=numClasses, bias=False)
		self.bottleneck = nn.BatchNorm1d(self.feat_dim)
		self.bottleneck.bias.requires_grad_(False)

	def forward(self, x, b, t):
		_, _, h, w = x.shape
		x = x.reshape(t, b, self.feat_dim, h, w)
		x = x.permute(1, 0, 2, 3, 4).reshape(-1, self.feat_dim, h, w) #[160, 192, 3, 2]

		a = F.relu(self.attention_conv(x))
		# print("a0.shape ================= ") #[160,128,1,1]
		# print(a.shape)
		a = a.reshape(b, t, self.middle_dim)
		# print("a1.shape ================= ") #[32,5,128]
		# print(a.shape)
		a = a.permute(0,2,1) #[32,128,5]
		a = F.relu(self.attention_tconv(a))
		# print("a2.shape ================= ") #[32,1,5]
		# print(a.shape)
		a = a.reshape(b, t) #[32,5]
		
		
		x = F.avg_pool2d(x, x.size()[2:]) #[160, 192, 1, 1]
		# print("xxxxxxxxxxx.shape ================= ") #[32,5]
		# print(x.shape)
		
		a = F.softmax(a, dim=1)
		x = x.reshape(b, t, -1) #[32, 5, 192]
		# print("a1.shape ================= ") #[32,5]
		# print(a.shape)
		a = torch.unsqueeze(a, -1)
		# print("a2.shape ================= ") #[32,5,1]
		# print(a.shape)
		a = a.expand(b, t, self.feat_dim)
		# print("a3.shape ================= ") #[32,5,192]
		# print(a.shape)
		# print("x.shape ================= ")  #[32,5,192]
		# print(x.shape)
		att_x = torch.mul(x, a)
		# print("att_x.shape ================= ")  #[32,5,192]
		# print(att_x.shape)
		att_x = torch.sum(att_x, 1)
		# print("att_x.shape ================= ")  #[32,192]
		# print(att_x.shape)
		

		att_x = self.bottleneck(att_x)
		out = self.fc(att_x)
		
		
		return att_x, out


class channelAttention(nn.Module):
	def __init__(self, featDim):
		super(channelAttention, self).__init__()
		self.feat_dim = featDim # feature dimension
		self.middle_dim = 1024 # middle layer dimension
		self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [1, 1])
		self.attention_tconv = nn.Conv2d(self.middle_dim, 1, [1, 5])

	def forward(self, x, b, t):
		bt, hw, f = x.shape
		x = x.permute(0, 2, 1).reshape(bt, f, 3, 2) #[160, 2048, 3, 2]
		
		a = F.relu(self.attention_conv(x)) #[160 ,1024, 3, 2]
		a = a.reshape(t, b, self.middle_dim, hw) #[5, 32, 1024, 6]
		a = a.permute(1, 2, 3, 0) #[32, 1024, 6, 5]
		a = F.relu(self.attention_tconv(a)) #[32, 1, 6, 1]
		a = a.reshape(b, -1) #[32,6]
		
		x = x.reshape(t, b, f, hw).permute(1, 2, 0, 3) #[32, 2048, 5, 6]

		x = F.avg_pool2d(x, [x.size()[2], 1]) #x.shape = [32, 2048, 1, 6]
		
		a = F.softmax(a, dim=1)  #[32,6]
		x = x.reshape(b, f, -1) #x.shape = [32, 2048, 6]
		x = x.permute(0, 2, 1) #[32, 6, 2048]
		a = torch.unsqueeze(a, -1) #[32, 6, 1]
		a = a.expand(b, hw, f) #[32, 6, 2048]
		att_x = torch.mul(x, a) #x.shape = a.shape = [32, 6, 2048]
		att_x = torch.sum(att_x, 1) #[32, 2048]
		
		return att_x


class temporalAttention(nn.Module):
	def __init__(self, featDim):
		super(temporalAttention, self).__init__()
		self.feat_dim = featDim # feature dimension
		self.middle_dim = 1024 # middle layer dimension
		self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [3, 2])
		self.attention_tconv = nn.Conv1d(self.middle_dim, 1, 3, padding=1)

	def forward(self, x, b, t):
		bt, hw, f = x.shape
		x = x.permute(0, 2, 1).reshape(bt, f, 3, 2) #[160, 2048, 3, 2]
		
		a = F.relu(self.attention_conv(x)) #[160 ,1024, 1, 1]
		a = a.reshape(t, b, self.middle_dim) #[5, 32, 1024]
		a = a.permute(1, 2, 0) #[32, 1024, 5]
		a = F.relu(self.attention_tconv(a)) #[32, 1, 5]
		a = a.reshape(b, -1) #[32, 5]
		
		x = F.avg_pool2d(x, x.size()[2:]) #x.shape = [160, 2048, 1, 1]

		x = x.reshape(t, b, f, 1).permute(1, 0, 2, 3) #[32, 5, 2048, 1]
		
		a = F.softmax(a, dim=1)  #[32, 5]
		x = x.reshape(b, 5, -1) #x.shape = [32, 5, 2048]
		a = torch.unsqueeze(a, -1) #[32, 5, 1]
		a = a.expand(b, 5, f) #[32, 5, 2048]
		att_x = torch.mul(x, a) #x.shape = a.shape = [32, 5, 2048]
		att_x = torch.sum(att_x, 1) #[32, 2048]
		
		return att_x



class domain2Vector(nn.Module):
	def __init__(self, csiChannel, numIds, featDim):
		super(domain2Vector, self).__init__()

		self.Encoder = encoder(csiChannel, featDim)

		self.idAttention = channelAttention(featDim=2048)
		self.actionAttention = channelAttention(featDim=2048)

		self.DisentanglerCommon = nn.Sequential(
			nn.Linear(in_features=32*6, out_features=3072, bias=False),
			nn.BatchNorm1d(6),
			nn.ReLU(inplace = True)
		)

		self.DisentanglerId = nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(in_features=3072, out_features=2048, bias=False),
			nn.BatchNorm1d(6),
			nn.ReLU(inplace = True)
		)

		self.DisentanglerAction = nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(in_features=3072, out_features=2048, bias=False),
			nn.BatchNorm1d(6),
			nn.ReLU(inplace = True)
		)

		self.IdDomainClassifier = nn.Sequential(
			nn.Linear(in_features=2048, out_features=256, bias=False),
			nn.ReLU(inplace = True),
			nn.Linear(in_features=256, out_features=50, bias=False)
		)

		self.ActDomainClassifier = nn.Sequential(
			nn.Linear(in_features=2048, out_features=256, bias=False),
			nn.ReLU(inplace = True),
			nn.Linear(in_features=256, out_features=4, bias=False)
		)


	def weights_init(self):
		for m in self.modules():
			if isinstance(m, (nn.Conv2d, nn.Linear, nn.Conv3d)):
				nn.init.xavier_normal_(m.weight)
			elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)


	def forward(self, x1, x2=0):
		b, t, c, f, h, w = x1.size()
		x1 = x1.permute(1, 0, 2, 3, 4, 5).reshape(-1, c, f, h, w)
		# x2 = x2.permute(1, 0, 2, 3, 4, 5).reshape(-1, c, f, h, w)

		# Encoder
		x1IdEncoder = self.Encoder(x1)
		# x2IdEncoder = self.Encoder(x2)
		bt, c, h, w = x1IdEncoder.size()
		x11IdEncoder = x1IdEncoder.reshape(bt, c, -1).permute(0, 2, 1)
		# x22IdEncoder = x2IdEncoder.reshape(bt, c, -1).permute(0, 2, 1)

		# Disentangle
		# x1MidDisentangler.shpae = [160, 6, 3072]
		# x1IdFeat.shape = x1ActFeat.shape = [160, 6, 2048]
		x1MidDisentangler = self.DisentanglerCommon(x11IdEncoder)
		x1IdFeat = self.DisentanglerId(x1MidDisentangler)
		x1ActFeat = self.DisentanglerAction(x1MidDisentangler)
		# x2MidDisentangler = self.DisentanglerCommon(x22IdEncoder)
		# x2IdFeat = self.DisentanglerId(x2MidDisentangler)
		# x2ActFeat = self.DisentanglerAction(x2MidDisentangler)

		# Classifier
		# x1idc.shape = [32, 2048]
		# x1actc.shape = [32, 2048]
		x1Idc = self.idAttention(x1IdFeat, b, t)
		x1Actc = self.actionAttention(x1ActFeat, b, t)
		# x2Idc = self.idAttention(x2IdFeat, b, t)
		# x2Actc = self.actionAttention(x2ActFeat, b, t)

		# Domain classifier
		# x1IdDomain.shape = [32, 50]
		# x1ActDomain.shape = [32, 4]
		x1IdDomain = self.IdDomainClassifier(x1Idc)
		x1ActDomain = self.ActDomainClassifier(x1Actc)
		# x2IdDomain = self.IdDomainClassifier(x2Idc)
		# x2ActDomain = self.ActDomainClassifier(x2Actc)


		return x1Idc, x1IdDomain, x1ActDomain
		# return x1Idc, x1IdDomain, x1ActDomain, x2Idc, x2IdDomain, x2ActDomain


class disentangle(nn.Module):
	def __init__(self, csiChannel, numIds, featDim):
		super(disentangle, self).__init__()

		self.idEncoder = encoder(csiChannel, featDim)
		self.actionEncoder = encoder(csiChannel, featDim)
		
		self.idClassifier = discriminator(numClasses=numIds)
		self.actionClassifier = discriminator(numClasses=4)

		self.generator = nn.Sequential(
			ConvBlock(192, 256, 3, 1),
			ConvBlock(256, csiChannel*25, 3, 1),
			nn.Conv2d(csiChannel*25, csiChannel*25, kernel_size=3, stride=1, padding=1, bias=False)
		)


	def weights_init(self):
		for m in self.modules():
			if isinstance(m, (nn.Conv2d, nn.Linear, nn.Conv3d)):
				nn.init.xavier_normal_(m.weight)
			elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)


	def forward(self, x1, x2=0):
		if self.training:
			b, t, c, f, h, w = x1.size()
			x1 = x1.permute(1, 0, 2, 3, 4, 5).reshape(-1, c, f, h, w)
			x2 = x2.permute(1, 0, 2, 3, 4, 5).reshape(-1, c, f, h, w)
			print("x1 shape ======================================= ")
			print(x1.shape)
			x1IdEncoder = self.idEncoder(x1)
			x1ActionFeature = self.actionEncoder(x1)
			x2IdEncoder = self.idEncoder(x2)
			x2ActionFeature = self.actionEncoder(x2)
			print(x1IdEncoder.shape)
			exit()

			x1IdFeat, x1Id = self.idClassifier(x1IdEncoder, b, t)
			_, x1Action = self.actionClassifier(x1ActionFeature, b, t)
			x2IdFeat, x2Id = self.idClassifier(x2IdEncoder, b, t)
			_, x2Action = self.actionClassifier(x2ActionFeature, b, t)

			x1Id_x2Action = self.generator(torch.cat((x1IdEncoder, x2ActionFeature), dim=1))
			x1Id_x2Action = x1Id_x2Action.reshape(x1Id_x2Action.size()[0], c, f, h, w)
			x2Id_x1Action = self.generator(torch.cat((x2IdEncoder, x1ActionFeature), dim=1))
			x2Id_x1Action = x2Id_x1Action.reshape(x2Id_x1Action.size()[0], c, f, h, w)

			x1IdEncoder_ = self.idEncoder(x1Id_x2Action)
			x2ActionFeature_ = self.actionEncoder(x1Id_x2Action)
			x2IdEncoder_ = self.idEncoder(x2Id_x1Action)
			x1ActionFeature_ = self.actionEncoder(x2Id_x1Action)

			x1IdFeat_, x1Id_ = self.idClassifier(x1IdEncoder_, b, t)
			_, x1Action_ = self.actionClassifier(x1ActionFeature_, b, t)
			x2IdFeat_, x2Id_ = self.idClassifier(x2IdEncoder_, b, t)
			_, x2Action_ = self.actionClassifier(x2ActionFeature_, b, t)

			x1Id_x1Action = self.generator(torch.cat((x1IdEncoder_, x1ActionFeature_), dim=1))
			x1Id_x1Action = x1Id_x1Action.reshape(x1Id_x1Action.size()[0], c, f, h, w)
			x2Id_x2Action = self.generator(torch.cat((x2IdEncoder_, x2ActionFeature_), dim=1))
			x2Id_x2Action = x2Id_x2Action.reshape(x2Id_x2Action.size()[0], c, f, h, w)

			return (x1Id, x1Action, x2Id, x2Action, x1IdFeat, x2IdFeat, x1IdEncoder, x2IdEncoder),\
					(x1Id_, x1Action_, x2Id_, x2Action_, x1Id_x1Action, x2Id_x2Action, x1IdFeat_, x2IdFeat_, x1IdEncoder_, x2IdEncoder_)

		else:
			b, t, c, f, h, w = x1.size()
			x1 = x1.permute(1, 0, 2, 3, 4, 5).reshape(-1, c, f, h, w)

			x1IdFeature = self.idEncoder(x1)
			x1ActionFeature = self.actionEncoder(x1)

			x1IdFeat, x1Id = self.idClassifier(x1IdFeature, b, t)
			_, x1Action = self.actionClassifier(x1ActionFeature, b, t)

			return x1Id, x1IdFeat


class disentanglePhase(nn.Module):
	def __init__(self, csiChannel, numIds, featDim):
		super(disentanglePhase, self).__init__()

		self.idEncoder = encoder(csiChannel, featDim)
		self.actionEncoder = encoder(csiChannel, featDim)
		
		self.idClassifier = discriminatorPhase(numClasses=numIds, featDim=featDim*6)
		self.actionClassifier = discriminatorPhase(numClasses=4, featDim=featDim*6)

		self.generator = nn.Sequential(
			ConvBlock(featDim*12, 256, 3, 1),
			ConvBlock(256, csiChannel*25, 3, 1),
			nn.Conv2d(csiChannel*25, csiChannel*25, kernel_size=3, stride=1, padding=1, bias=False)
		)


	def weights_init(self):
		for m in self.modules():
			if isinstance(m, (nn.Conv2d, nn.Linear, nn.Conv3d)):
				nn.init.xavier_normal_(m.weight)
			elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)


	def forward(self, x1, x2=0):
		if self.training:
			b, t, c, f, h, w = x1.size()
			x1 = x1.permute(1, 0, 2, 3, 4, 5).reshape(-1, c, f, h, w)
			x2 = x2.permute(1, 0, 2, 3, 4, 5).reshape(-1, c, f, h, w)
			x1IdEncoder = self.idEncoder(x1)
			x1ActionFeature = self.actionEncoder(x1)
			x2IdEncoder = self.idEncoder(x2)
			x2ActionFeature = self.actionEncoder(x2)

			x1IdFeat, x1Id = self.idClassifier(x1IdEncoder, b, t)
			_, x1Action = self.actionClassifier(x1ActionFeature, b, t)
			x2IdFeat, x2Id = self.idClassifier(x2IdEncoder, b, t)
			_, x2Action = self.actionClassifier(x2ActionFeature, b, t)

			x1Id_x2Action = self.generator(torch.cat((x1IdEncoder, x2ActionFeature), dim=1))
			x1Id_x2Action = x1Id_x2Action.reshape(x1Id_x2Action.size()[0], c, f, h, w)
			x2Id_x1Action = self.generator(torch.cat((x2IdEncoder, x1ActionFeature), dim=1))
			x2Id_x1Action = x2Id_x1Action.reshape(x2Id_x1Action.size()[0], c, f, h, w)

			x1IdEncoder_ = self.idEncoder(x1Id_x2Action)
			x2ActionFeature_ = self.actionEncoder(x1Id_x2Action)
			x2IdEncoder_ = self.idEncoder(x2Id_x1Action)
			x1ActionFeature_ = self.actionEncoder(x2Id_x1Action)

			x1IdFeat_, x1Id_ = self.idClassifier(x1IdEncoder_, b, t)
			_, x1Action_ = self.actionClassifier(x1ActionFeature_, b, t)
			x2IdFeat_, x2Id_ = self.idClassifier(x2IdEncoder_, b, t)
			_, x2Action_ = self.actionClassifier(x2ActionFeature_, b, t)

			x1Id_x1Action = self.generator(torch.cat((x1IdEncoder_, x1ActionFeature_), dim=1))
			x1Id_x1Action = x1Id_x1Action.reshape(x1Id_x1Action.size()[0], c, f, h, w)
			x2Id_x2Action = self.generator(torch.cat((x2IdEncoder_, x2ActionFeature_), dim=1))
			x2Id_x2Action = x2Id_x2Action.reshape(x2Id_x2Action.size()[0], c, f, h, w)
			
			return (x1Id, x1Action, x2Id, x2Action, x1IdFeat, x2IdFeat, x1IdEncoder, x2IdEncoder),\
					(x1Id_, x1Action_, x2Id_, x2Action_, x1Id_x1Action, x2Id_x2Action, x1IdFeat_, x2IdFeat_, x1IdEncoder_, x2IdEncoder_)

		else:
			b, t, c, f, h, w = x1.size()
			x1 = x1.permute(1, 0, 2, 3, 4, 5).reshape(-1, c, f, h, w)

			x1IdFeature = self.idEncoder(x1)
			x1ActionFeature = self.actionEncoder(x1)

			x1IdFeat, x1Id = self.idClassifier(x1IdFeature, b, t)
			_, x1Action = self.actionClassifier(x1ActionFeature, b, t)

			return x1Id, x1IdFeat


class baseline(nn.Module):
	def __init__(self, csiChannel, numIds, featDim):
		super(baseline, self).__init__()
		self.idEncoder = encoder(csiChannel, featDim)
		self.idClassifier = classifier(numClasses=numIds)

	def weights_init(self):
		for m in self.modules():
			if isinstance(m, (nn.Conv2d, nn.Linear, nn.Conv3d)):
				nn.init.xavier_normal_(m.weight)
			elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x1):
		b, t, c, f, h, w = x1.size()
		x1 = x1.permute(1, 0, 2, 3, 4, 5).reshape(-1, c, f, h, w)

		x1IdFeature = self.idEncoder(x1)
		x1IdFeat, x1Id = self.idClassifier(x1IdFeature, b, t)

		return x1Id, x1IdFeat