#coding=utf-8
'''
Source code for an attention based image caption generation system described
in:
Show, Attend and Tell: Neural Image Caption Generation with Visual Attention
International Conference for Machine Learning (2015)
http://arxiv.org/abs/1502.03044
'''

import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from darknet import Darknet
import numpy as np

MAX_LENGTH = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cuda_variable(tensor):
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)

class EncoderCNN(nn.Module):
    def __init__(self,reso, cuda=torch.cuda.is_available()):
        super(EncoderCNN, self).__init__()


        self.model = Darknet("cfg/yolov3.cfg")
        self.model.load_weights("yolov3.weights")
        self.CUDA=cuda
        self.model.net_info["height"] = reso
    def forward(self, images):
        inp_dim = int(self.model.net_info["height"])
        assert inp_dim % 32 == 0
        assert inp_dim > 32
        prediction, feature13, feature26, feature52 = self.model(images, self.CUDA)
        #feature13 = feature13.permute(0, 2, 3, 1)

        return prediction, feature13, feature26, feature52


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out,decoder_hidden):
        #encoder_out=(32*196*2048),对应的是图像的特征
	    #decoder_hidden是隐层状态h(32*512)
        #att1=(32*196*512)
        att1 = self.encoder_att(encoder_out)
        #att2=(32*512)
        att2 = self.decoder_att(decoder_hidden)
        #self.relu(att1 + att2.unsqueeze(1))计算完的结果是(32*196*512)
        #att=(32*196),att是图像的特征和h相加而得，对每个feature map的196个值求权重，而后2048个feature均乘以这个权重
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        #对att的dim=1进行计算权重，相当于对每个image的196个特征求权重，维度大小不变，仍等同于att
        alpha = self.softmax(att)
        #encoder_out * alpha.unsqueeze(2)的结果是(32*196*2048)
        #attention_weighted_encoding=(32*2048)，将同一个feature map的196个值相加，最后得到的是2048个值，每个值代表一个feature map
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return attention_weighted_encoding, alpha

class AttnDecoderRNN(nn.Module):

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab,filename,encoder_dim=255, dropout=0.5):
        super(AttnDecoderRNN, self).__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = len(vocab)
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        self.embedding = self.load_vec(vocab,embed_dim,filename)
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)   # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, self.vocab_size)
        self.init_weights()

    def load_vec(self,vocab,embed_dim,filename):
        embedding=nn.Embedding(len(vocab), embed_dim)
        vec_dict = {}
        with open(filename,'r') as f:
            lines=f.readlines()
            for line in lines:
                line=line.split()
                vec_dict[line[0]]=line[1:]
        for key,value in vocab:
            if key in vec_dict:
                embedding.weight[value,:]=vec_dict[key]
        return embedding

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, box_feature, encoded_captions, caption_lengths):
        """
        :arg encoder_out:32*14*14*2048,encoded_captions:32*21,caption_length:list and the length is 32
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights
        """
        box_feature=box_feature.permute(0,2,1)
        encoder_out= encoder_out.permute(0, 2, 3, 1)
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        #vocab_size=9956
        vocab_size = self.vocab_size
        #encoder_out=(32*196*2048)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        encoder_out=torch.cat((encoder_out,box_feature),dim=1)
        num_pixels = encoder_out.size(1)
        #embeddings=(32*21*512)
        embeddings = self.embedding(encoded_captions)
        #h=(32*512)
        h, c = self.init_hidden_state(encoder_out)

        decode_lengths = [c-1 for c in caption_lengths]
        #predictions=(32*20*9956)
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        #alphas=(32*20*196)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)
	    #如下循环是将一批数据，这批数据有32个句子，按照最长的句子循环，同时生成32个句子的每一个单词
        for t in range(max(decode_lengths)):
		    #batch_size_t是控制那些每次循环那些句子生成单词，但句子短时，到最后的几次循环，短句子就不生成单词
            batch_size_t = sum([l > t for l in decode_lengths ])
            #attention_weighted_encoding=(32*2048),alpha=(32*196)
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            #gate=(32*2048)
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            # attention_weighted_encoding=(32*2048)
            attention_weighted_encoding = gate * attention_weighted_encoding
            #将caption和加权重之后的image特征拼接作为input输入，而后再将h和c输入到decode_step中
            h, c = self.decode_step(
                torch.cat((embeddings[:batch_size_t, t, :], attention_weighted_encoding), dim=1),
                (h[:batch_size_t], c[:batch_size_t]))
            #preds=(32*9956)，fc全连接层，然后通过h将结果输出
            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas
