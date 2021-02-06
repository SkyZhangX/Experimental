# -*- coding: utf-8 -*-


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
import nltk
class configs:
    epoch = 50
    batch_size = 10
    lr = 0.01
    decay = 0
    LTrain = 1000
    LTest = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        #output = embedded
        output, hidden = self.gru(embedded, hidden)

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
       
    def forward(self, input, hidden):

        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])

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
        #initialize a list of output, matching input dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        en_outs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        #encoder output to context vector
        
        for ei in range(trg_len):
            if src[ei] !=-1:
                en_out,hidden = self.encoder(src[ei],hidden)
   
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
            if trg[t]!= -1:
                loss += criterion(output,trg[t])
            #0 for use real output, 1 for use best model output 
            input = trg[t] if random.random() > teacher_forcing_ratio else best
        
        return outputs,loss



def char2vec(word_list,vec = {'\t':0,'\n':1,'PAD':-1}):
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
    for data in new_data:
        while len(data)<57:
            data.append('PAD')
    return new_data

def load_src_data():
    data_file = 'padding.txt'
   
    data_f = open(data_file,'r').readlines()
    data_tokenization = tokenize(data_f)
    encode, vec = char2vec(data_tokenization)

    trg_file = 'tpadding.txt'
    trg_f = open(trg_file,'r').readlines()
    trg_tokenization = tokenize(trg_f)
    decode, _   = char2vec(trg_tokenization,vec)
    train_length = configs.LTrain
    test_length = configs.LTest
    
    return (encode[:train_length], decode[:train_length]),(encode[train_length:train_length+test_length], decode[train_length:train_length+test_length]), vec

(data_train, trg_train),(data_test,trg_test), vec = load_src_data()
#print(data_train[:10],trg_train[:10])

cat_training = torch.cat((data_train.unsqueeze(0),trg_train.unsqueeze(0)),dim=0)
cat_test =  torch.cat((data_test.unsqueeze(0),trg_test.unsqueeze(0)),dim=0)



INPUT_DIM = len(vec)
OUTPUT_DIM = len(vec)

HID_DIM = 280
N_LAYERS = 1
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1


enc = EncoderRNN(INPUT_DIM, HID_DIM).to(device)#, HID_DIM, N_LAYERS, ENC_DROPOUT)
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

            texts = texts.squeeze(1)
            trg = trg.squeeze(1)
            '''
            if epoch%10==0 and index in [1,2,3]:
                print('/n')
                print(epoch)
                print(('pred,',output.argmax(1)),('trg',trg))
            '''
            #loss = criterion(output,trg)
            loss.backward()
            optimizer.step()
        if epoch%10==0:
          print(total_loss/config.LTrain)
        
        #print('\n Validating...')
        if (epoch+1)%10==0:
            print('\n Validating...')
            validate(eval_data,criterion,best_acc)
        losses.append(loss)


def validate(eval_data,criterion,best_acc):
    model.eval()
    total_loss = 0
    epoch_loss = 0
    mean_acc = 0
    total = 0
    correct = 0
    bleu_score = 0
    with torch.no_grad():
        for index, (texts,trg) in tqdm(enumerate(eval_data),total=configs.LTest,position=0,leave=True):
            
            texts = texts.unsqueeze(1).to(device)
            trg = trg.unsqueeze(1).to(device)
           
            output,loss = model(texts,trg)
            total_loss +=loss.item()
            output_dim = output.shape[-1]
            output = output[0:].view(-1,output_dim)
            trg = trg.squeeze(1)

            pred =  output.argmax(1)
           
            
            for i,v in enumerate(trg):
                if v == -1:
                    padIndex = i
                    break
            #print(pred[:padIndex],trg[:padIndex])
            correct += int(torch.equal(pred[:padIndex],trg[:padIndex]))#.sum().item()
            #print(correct)
           
            if index in [13,15,17]:
                print(('pred,',pred[:15]),('trg',trg[:15]))
            
            temp_pred = []
            temp_trg = []
            for i in range(len(pred)):
                if trg[i]!=-1:
                    temp_pred.append(str(pred[i]))
                    temp_trg.append(str(trg[i]))
            bleu_score += nltk.translate.bleu_score.modified_precision([temp_trg], temp_pred,1)
        
        total = configs.LTest
        print(correct,total)
        print('Loss is: ',total_loss/configs.LTest)
        mean_acc = correct / total
        print("Pred accuracy",mean_acc)
        print("BLEU accuracy",float(bleu_score/total))

    if mean_acc >best_acc :
        print("=================================="\
                    "Current Model Saved"\
              "==================================")

        if not os.path.isdir('checkpoint'):
          os.mkdir('checkpoint')
        torch.save(model.state_dict(), './checkpoint/ckpt.pth')
        best_acc = mean_acc
    return

training(torch.transpose(cat_training,0,1),torch.transpose(cat_test,0,1))