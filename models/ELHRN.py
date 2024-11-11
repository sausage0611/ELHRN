import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as torch_models
import numpy as np
from .backbones import Conv_4, ResNet
from .backbones.HDM import HDM
from .backbones.DCM import DCM
from .backbones.SWTM import SWTM



class ELHRN(nn.Module):

    def __init__(self, way=None, shots=None, resnet=False):

        super().__init__()

        self.resolution = 5 * 5
        if resnet:
            self.num_channel = 640
            self.feature_extractor = ResNet.resnet12()
            self.dim = self.num_channel * 5 * 5

        else:
            self.num_channel = 64
            self.feature_extractor = Conv_4.BackBone(self.num_channel)
            self.dim = self.num_channel * 5 * 5

        self.fsrm = HDM(
            sequence_length=self.resolution,
            embedding_dim=self.num_channel,
            num_layers=2,
            num_heads=2,
            mlp_dropout_rate=0.,
            attention_dropout=0.,
            positional_embedding='sine')

        self.fmrm1 = DCM(hidden_size=self.num_channel, inner_size=self.num_channel, num_patch=self.resolution,
                         drop_prob=0.1)
        self.fmrm2 = DCM(hidden_size=self.num_channel, inner_size=self.num_channel, num_patch=self.resolution,
                         drop_prob=0.1)


        self.shots = shots
        self.way = way
        self.resnet = resnet

        self.scale = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

        self.w1 = nn.Parameter(torch.FloatTensor([0.25]), requires_grad=True)
        self.w2 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.w3 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.w4 = nn.Parameter(torch.FloatTensor([0.8]), requires_grad=True)

        self.autoParmFive = SWTM()


    def get_feature_vector(self, inp):
        batch_size = inp.size(0)
        feature_map = self.feature_extractor(inp)

        results = self.fsrm(feature_map)
        layer1_feature_map = results[0]
        layer2_feature_map = results[1]

        layer1_feature_map = layer1_feature_map.transpose(1, 2).view(batch_size, self.num_channel, 5, 5)
        layer2_feature_map = layer2_feature_map.transpose(1, 2).view(batch_size, self.num_channel, 5, 5)

        return feature_map,layer1_feature_map,layer2_feature_map

    def get_neg_l2_dist(self, inp, way, shot, query_shot):
        feature_map, layer1_feature_map, layer2_feature_map = self.get_feature_vector(inp)

        layer1_support = layer1_feature_map[:way * shot].view(way, shot, *layer1_feature_map.size()[1:]).permute(0, 2, 1, 3,
                                                                                            4).contiguous()
        layer1_query = layer1_feature_map[way * shot:]


        layer2_support = layer2_feature_map[:way * shot].view(way, shot, *layer2_feature_map.size()[1:]).permute(0, 2, 1, 3,
                                                                                            4).contiguous()
        layer2_query = layer2_feature_map[way * shot:]

        layer1_sq_similarity, layer1_qs_similarity = self.fmrm1(layer1_support, layer1_query)
        layer2_sq_similarity, layer2_qs_similarity = self.fmrm2(layer2_support, layer2_query)




        #train_second
        # output = self.autoParmFive(way, shot)
        # l1 = output[0, 0]  # 注意这里使用逗号分隔索引，而不是两个方括号
        # l2 = output[0, 1]
        # l3 = output[0, 2]
        # l4 = output[0, 3]
        # l2_dist = l1 * self.w1 * layer1_sq_similarity + l2 * self.w2 * layer1_qs_similarity + l3 * self.w3 * layer2_sq_similarity + l4 * self.w4* layer2_qs_similarity


        #train_first
        l2_dist = self.w1 * layer1_sq_similarity + self.w2 * layer1_qs_similarity + self.w3 * layer2_sq_similarity + self.w4 * layer2_qs_similarity




        return l2_dist

    def meta_test(self, inp, way, shot, query_shot):

        neg_l2_dist = self.get_neg_l2_dist(inp=inp,
                                           way=way,
                                           shot=shot,
                                           query_shot=query_shot)

        _, max_index = torch.max(neg_l2_dist, 1)

        return max_index


    #train_second
    # def forward(self, inp, way, shot, query_shot):
    #
    #     logits = self.get_neg_l2_dist(inp=inp,
    #                                   way=way,
    #                                   shot=shot,
    #                                   query_shot=query_shot)
    #     logits = logits / self.dim * self.scale
    #     #
    #     log_prediction = F.log_softmax(logits, dim=1)
    #
    #     return log_prediction

    #train_first
    def forward(self, inp):

        logits = self.get_neg_l2_dist(inp=inp,
                                      way=self.way,
                                      shot=self.shots[0],
                                      query_shot=self.shots[1])
        logits = logits / self.dim * self.scale

        log_prediction = F.log_softmax(logits, dim=1)

        return log_prediction