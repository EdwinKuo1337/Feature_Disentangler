from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision
from .backbones.resnet import ResNet, BasicBlock, Bottleneck
import math

__all__ = ['ResNet50TP', 'ResNet50TA', 'ResNet50RNN', 'ResNet50TAPhase']


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class ResNet50TAPhase(nn.Module):
    def __init__(self, num_classes, neck='bnneck', neck_feat='after', **kwargs):
        super(ResNet50TAPhase, self).__init__()
        # resnet50 = torchvision.models.resnet50(pretrained=True)
        # self.base1 = nn.Sequential(*list(resnet50.children())[:-2])
        self.base = ResNet(last_stride=2, block=Bottleneck, layers=[3, 4, 6, 3])
        self.att_gen = 'softmax' # method for attention generation: softmax or sigmoid
        self.feat_dim = 2048 # feature dimension
        self.middle_dim = 256 # middle layer dimension
        # self.classifier = nn.Linear(self.feat_dim, num_classes)
        self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [7,4]) # 7,4 cooresponds to 224, 112 input image size
        self.attention_tconv = nn.Conv1d(self.middle_dim, 1, 3, padding=1)
        self.neck = neck
        self.neck_feat = neck_feat
        self.num_classes = num_classes

        self.Convolution3D_1 = nn.Conv3d(30, 30, kernel_size=(16, 3, 3), padding=(0, 1, 1))
        self.Convolution3D_2 = nn.Conv3d(30, 30, kernel_size=(10, 3, 3), padding=(0, 1, 1))


        if self.neck == 'no':
            self.classifier = nn.Linear(self.feat_dim, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.feat_dim)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.feat_dim, self.num_classes, bias=False)
            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)


    def forward(self, x):
        b, t, c, f, h, w = x.size()
        x = x.view(b*t, c, f, h, w)
        # x = x.permute(0, 2, 1, 3, 4)

        x = self.Convolution3D_1(x)
        x = self.Convolution3D_2(x)

        x = x.squeeze(2)
        x = F.interpolate(x, size=[224, 112], mode='bilinear', align_corners=False)

        # x = x.view(b*t, x.size(2), x.size(3), x.size(4))
        x = self.base(x)

        a = F.relu(self.attention_conv(x))
        a = a.view(b, t, self.middle_dim)
        a = a.permute(0,2,1)
        a = F.relu(self.attention_tconv(a))
        a = a.view(b, t)
        x = F.avg_pool2d(x, x.size()[2:])
        if self. att_gen=='softmax':
            a = F.softmax(a, dim=1)
        elif self.att_gen=='sigmoid':
            a = F.sigmoid(a)
            a = F.normalize(a, p=1, dim=1)
        else: 
            raise KeyError("Unsupported attention generation function: {}".format(self.att_gen))
        
        x = x.view(b, t, -1)
        a = torch.unsqueeze(a, -1)
        a = a.expand(b, t, self.feat_dim)
        att_x = torch.mul(x,a)
        att_x = torch.sum(att_x,1)
        global_feat = att_x.view(b,self.feat_dim)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax


        y = self.classifier(feat)
        return y, feat
        
        # if self.training:
        #     y = self.classifier(feat)
        #     return y, feat
        # else:
        #     if self.neck_feat == 'after':
        #         # print("Test with feature after BN")
        #         return feat
        #     else:
        #         # print("Test with feature before BN")
        #         return global_feat

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ResNet50TA(nn.Module):
    def __init__(self, num_classes, neck='bnneck', neck_feat='after', **kwargs):
        super(ResNet50TA, self).__init__()
        # resnet50 = torchvision.models.resnet50(pretrained=True)
        # self.base1 = nn.Sequential(*list(resnet50.children())[:-2])
        self.base = ResNet(last_stride=2, block=Bottleneck, layers=[3, 4, 6, 3])
        self.att_gen = 'softmax' # method for attention generation: softmax or sigmoid
        self.feat_dim = 2048 # feature dimension
        self.middle_dim = 256 # middle layer dimension
        # self.classifier = nn.Linear(self.feat_dim, num_classes)
        self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [7,4]) # 7,4 cooresponds to 224, 112 input image size
        self.attention_tconv = nn.Conv1d(self.middle_dim, 1, 3, padding=1)
        self.neck = neck
        self.neck_feat = neck_feat
        self.num_classes = num_classes

        self.Convolution3D_1 = nn.Conv3d(30, 30, kernel_size=(16, 3, 3), padding=(0, 1, 1))
        self.Convolution3D_2 = nn.Conv3d(30, 30, kernel_size=(10, 3, 3), padding=(0, 1, 1))


        if self.neck == 'no':
            self.classifier = nn.Linear(self.feat_dim, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.feat_dim)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.feat_dim, self.num_classes, bias=False)
            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)


    def forward(self, x):
        b, t, c, f, h, w = x.size()
        x = x.view(b*t, c, f, h, w)
        # x = x.permute(0, 2, 1, 3, 4)

        x = self.Convolution3D_1(x)
        x = self.Convolution3D_2(x)

        x = x.squeeze(2)
        x = F.interpolate(x, size=[224, 112], mode='bilinear', align_corners=False)

        # x = x.view(b*t, x.size(2), x.size(3), x.size(4))
        x = self.base(x)

        a = F.relu(self.attention_conv(x))
        a = a.view(b, t, self.middle_dim)
        a = a.permute(0,2,1)
        a = F.relu(self.attention_tconv(a))
        a = a.view(b, t)
        x = F.avg_pool2d(x, x.size()[2:])
        if self. att_gen=='softmax':
            a = F.softmax(a, dim=1)
        elif self.att_gen=='sigmoid':
            a = F.sigmoid(a)
            a = F.normalize(a, p=1, dim=1)
        else: 
            raise KeyError("Unsupported attention generation function: {}".format(self.att_gen))
        
        x = x.view(b, t, -1)
        a = torch.unsqueeze(a, -1)
        a = a.expand(b, t, self.feat_dim)
        att_x = torch.mul(x,a)
        att_x = torch.sum(att_x,1)
        global_feat = att_x.view(b,self.feat_dim)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax


        y = self.classifier(feat)
        return y, feat
        
        # if self.training:
        #     y = self.classifier(feat)
        #     return y, feat
        # else:
        #     if self.neck_feat == 'after':
        #         # print("Test with feature after BN")
        #         return feat
        #     else:
        #         # print("Test with feature before BN")
        #         return global_feat

            
    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ResNet50RNN(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50r, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.hidden_dim = 512
        self.feat_dim = 2048
        self.classifier = nn.Linear(self.hidden_dim, num_classes)
        self.lstm = nn.LSTM(input_size=self.feat_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)
    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b*t,x.size(2), x.size(3), x.size(4))
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(b,t,-1)
        output, (h_n, c_n) = self.lstm(x)
        output = output.permute(0, 2, 1)
        f = F.avg_pool1d(output, t)
        f = f.view(b, self.hidden_dim)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet50TP(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50TP, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.feat_dim = 2048
        self.classifier = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b*t,x.size(2), x.size(3), x.size(4))
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(b,t,-1)
        x=x.permute(0,2,1)
        f = F.avg_pool1d(x,t)
        f = f.view(b, self.feat_dim)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))