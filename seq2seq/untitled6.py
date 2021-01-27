# -*- coding: utf-8 -*-
"""Untitled6.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MM5DKTaRTkq-chkreTXk0XZEIbNjb_Gk
"""

import torch 
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

#import others
import re
import time
import math
import copy
import random
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import os

class configs:
    epoch = 100
    batch_size = 10
    lr = 0.01
    decay = 0
    LTrain = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        #output = self.softmax(self.out(output[0]))
        output = self.out(output[0])
        #output  = self.out(output[0])
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        '''
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        '''
    def forward(self, src, trg, teacher_forcing_ratio = 1):
        
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = len(vec)#self.decoder.output_dim
        hidden = self.encoder.initHidden()
        # cell = self.encoder.initHidden()
        #initialize a list of output, matching input dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        en_outs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        #encoder output to context vector
        
        for ei in range(7):
            #print(src[ei])
            en_out,hidden = self.encoder(src[ei],hidden)
            #en_outs[ei] = en_out[0,0]
        

        #output,hidden = self.encoder(src,hidden)#,cell)
        #<sos> first in 
        input = trg[0,:]
        
        #initialize <eos> of encoder, <sos> of decoder
        outputs[0,0,0] = 1
        criterion = nn.CrossEntropyLoss()
        #outputs[6,0,1] = 1
        loss = 0
        for t in range(1,trg_len):
            
            output, hidden= self.decoder(input, hidden)#, cell)
            outputs[t] = output
            best = output.argmax(1)
            
            loss += criterion(output,trg[t])
            #0 for use real output, 1 for use best model output 
            #print(trg[t])
            
            input = trg[t] if random.random() > teacher_forcing_ratio else best
            #sys.exit()
            '''
            if input[0] == 1:
                break
            '''
        '''
        print("\n")
        print(t)
        print(src,outputs.shape)
        print(outputs.argmax(2))
        '''
        
        return outputs,loss



def char2vec(word_list,vec = {'\t':0,'\n':1}):
    num = 2
    new_word_list = []
    for wl in word_list:
        con_word_list = []
        for c in wl:
            if c not in vec:
                vec[c] = num
                con_word_list.append(num)
                num+=1
            else:
                con_word_list.append(vec[c])
            
        new_word_list.append(con_word_list)

    return torch.tensor(new_word_list).to(device), vec

def tokenize(token: list) -> list:
    new_data = [list('\t'+tok) for tok in token]
    return new_data

def load_src_data():
    data_file = 'samples.txt'
   
    data_f = open(data_file,'r').readlines()
    data_tokenization = tokenize(data_f)
    encode, vec = char2vec(data_tokenization)

    trg_file = 'target.txt'
    trg_f = open(trg_file,'r').readlines()
    trg_tokenization = tokenize(trg_f)
    decode, _   = char2vec(trg_tokenization,vec)
    train_length = configs.LTrain
    
  
    return (encode[:train_length], decode[:train_length]),(encode[:train_length], decode[:train_length]), vec

(data_train, trg_train),(data_test,trg_test), vec = load_src_data()

cat_training = torch.cat((data_train.unsqueeze(0),trg_train.unsqueeze(0)),dim=0)
cat_test =  torch.cat((data_test.unsqueeze(0),trg_test.unsqueeze(0)),dim=0)



INPUT_DIM = len(vec)
OUTPUT_DIM = len(vec)
ENC_EMB_DIM = 280
DEC_EMB_DIM = 280
HID_DIM = 280
N_LAYERS = 1
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1


enc = EncoderRNN(INPUT_DIM, ENC_EMB_DIM).to(device)#, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = DecoderRNN(HID_DIM, OUTPUT_DIM).to(device)#, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)


losses = []


def training(data,eval_data):
    config = configs()
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr = config.lr , weight_decay = config.decay)
    best_acc = 0
   
    for epoch in range(config.epoch):
        model.train()
        total_loss = 0
       # print('\nDATA',data)
        for index, (texts,trg) in tqdm(enumerate(data),total = configs.LTrain,leave=True,position = 0):
        #for texts in data:
            texts = texts.unsqueeze(1).to(device)
            trg = trg.unsqueeze(1).to(device)
            #print(texts.shape)
            optimizer.zero_grad()
            output,loss = model(texts,trg)
            total_loss +=loss.item()
            output_dim = output.shape[-1]
            output = output[0:].view(-1,output_dim)
            #print(output.argmax(dim=1))
            texts = texts.squeeze(1)
            trg = trg.squeeze(1)
            if epoch%10==0 and index in [1,2,3]:
                print('/n')
                print(epoch)
                print(('pred,',output.argmax(1)),('trg',trg))
                
            
              
            #loss = criterion(output,trg)
            loss.backward()
            optimizer.step()
        if epoch%10==0:
          print(total_loss/config.LTrain)
        losses.append(loss)
        #validate(eval_data,criterion,best_acc)


def validate(eval_data,criterion,best_acc):
    model.eval()
    epoch_loss = 0
    mean_acc = 0
    total = 0
    correct = 0
    with torch.no_grad():
    
        for _, (texts,trg) in tqdm(enumerate(eval_data),total=configs.LTrain,position=0,leave=True):

            texts = texts.unsqueeze(1)
            trg = trg.unsqueeze(1)
           
            
            output = model(texts,trg,0)
            output_dim = output.shape[-1]
            output = output[0:].view(-1,output_dim)
            trg = trg.squeeze(1)

            pred =  output.argmax(1)
            lossE = criterion(output,trg)

            total += trg.size(0)
            correct += (pred==trg).sum().item()
            #print("pred",pred)
            #print("texts",texts)


            
            epoch_loss += lossE.item()
            mean_acc = correct / total
        print(mean_acc)

    if mean_acc >best_acc :
        print("=================================="\
                    "Current Model Saved"\
              "==================================")
        
        #save model to file
        if not os.path.isdir('checkpoint'):
          os.mkdir('checkpoint')
        torch.save(model.state_dict(), './checkpoint/ckpt.pth')
        best_acc = mean_acc


training(torch.transpose(cat_training,0,1),torch.transpose(cat_test,0,1))