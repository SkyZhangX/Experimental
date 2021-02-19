# -*- coding: utf-8 -*-

import torch 
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
    batch_size = 2
    lr = 0.01
    decay = 0
    LTrain = 1000
    LTest = 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size,batch_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.size =  0

    def forward(self, input, hidden,input_length):
        self.size = input.shape[0]
        self.gru = nn.GRU(self.hidden_size*input.shape[0],self.hidden_size).to(device)

        embedded = self.embedding(input).view( 1,self.batch_size,-1)
 
        #pack = nn.utils.rnn.pack_padded_sequence(embedded, input_length,enforce_sorted=False)

        output, hidden = self.gru(embedded, hidden)

        return output, hidden

    def initHidden(self):
        
        return torch.zeros(1,2, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, batch_size, output_size):
        super(DecoderRNN, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
       
    def forward(self, input, hidden):
        output = self.embedding(input).view(1, self.batch_size, -1)
        # print("\n output",output)
        # print('--------------------------------------------------------------------')
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        # print(output)
        
        output = self.out(output[0])

        return output, hidden

    def initHidden(self):
       
        return torch.zeros(1,2, self.hidden_size, device=device)
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
    def forward(self, src, trg, teacher_forcing_ratio = 0):
        
        batch_size = configs.batch_size
       
        trg_vocab_size = len(vec)#self.decoder.output_dim
        hidden = self.encoder.initHidden()
        #initialize a list of output, matching input dim
        src = src.T
        trg_len = src.shape[0]
        outputs = torch.zeros(trg_len, configs.batch_size, trg_vocab_size).to(self.device)
        en_outs = torch.zeros(trg_len, configs.batch_size, trg_vocab_size).to(self.device)
        #encoder output to context vector
       
        packed = packing(src)

        #_,hidden = self.encoder(src,hidden, packed)
       
        #<sos> first in 
        input = torch.ones(src.shape[1],dtype = torch.long).to(device)
        
        #initialize <eos> of encoder, <sos> of decoder
        outputs[0,0,0] = 1
        criterion = nn.CrossEntropyLoss()
        loss = 0
        count = 0 
        #print(src)
        for t in range(1,trg_len):
            output, hidden= self.decoder(input, hidden)
            outputs[t] = output 
            best = output.argmax(1)
            for tr in trg:
                #print(tr)
                mk = masking(tr)
                for j,single in enumerate(tr):
                    
                    if mk[j,t].item() == True:
                        #print(tr[j,t])
                        count+=1
                        loss += criterion(output[j,:].unsqueeze(0).type(torch.float64),tr[j,t].unsqueeze(0).long()).item()
                            
        
            #0 for use real output, 1 for use best model output 
            input = tr[:,t] if random.random() > teacher_forcing_ratio else best
        #print('count',count)
        return outputs,loss

'''
test
'''




def char2vec(word_list,vec = {'\t':1,'\n':2,'PAD':0}):
    num = 3
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

def tokenize(token: list,pad = 5) -> list:
    new_data = [list('\t'+tok) for tok in token]
    for data in new_data:
        while len(data)<pad+7:
            data.append('PAD')
    return new_data

def load_src_data(pad=5):
    data_file = 'data/padding.txt'
   
    data_f = open(data_file,'r').readlines()
    data_tokenization = tokenize(data_f,pad)
    encode, vec = char2vec(data_tokenization)

    trg_file = 'data/tpadding.txt'
    trg_f = open(trg_file,'r').readlines()
    trg_tokenization = tokenize(trg_f,pad)
    decode, _   = char2vec(trg_tokenization,vec)

    train_length = configs.LTrain
    test_length = configs.LTest
    
    
    return (encode[:train_length], decode[:train_length]),(encode[train_length:train_length+test_length], decode[train_length:train_length+test_length]), vec

def masking(seq):
    
    mask = seq != 0
    
    return mask

def batching(seq,size):
    train = DataLoader(seq,batch_size = size)
    
    return train

def packing(seq):
    seq2 = torch.transpose(seq,0,1)
    test = []
    for ten in seq2:
        for i,num in enumerate(ten):
            if num==0:
                test.append(i)
                break
    #print(torch.tensor(test))
    return test#torch.tensor(test,dtype=torch.int).to(torch.device('cpu'))


(data_train, trg_train),(data_test,trg_test), vec = load_src_data()
src_mask,trg_mask = (data_train),masking(trg_train)

cat_training = torch.cat((data_train.unsqueeze(0),trg_train.unsqueeze(0)),dim=0)

cat_test =  torch.cat((data_test.unsqueeze(0),trg_test.unsqueeze(0)),dim=0)

src_batch,trg_batch = batching(data_train,configs.batch_size),batching(trg_train,configs.batch_size)

src_mask_batch, trg_mask_batch = batching(src_mask,configs.batch_size),batching(trg_mask,configs.batch_size)

test_src_batch,test_trg_batch = batching(data_test,configs.batch_size),batching(trg_test,configs.batch_size)

train_batch,test_batch  = batching(cat_training,configs.batch_size),batching(cat_test,configs.batch_size)

INPUT_DIM = len(vec)
OUTPUT_DIM = len(vec)

HID_DIM = 280
N_LAYERS = 1
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
BATCH = configs.batch_size

enc = EncoderRNN(INPUT_DIM, BATCH, HID_DIM).to(device)#, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = DecoderRNN(HID_DIM, BATCH, OUTPUT_DIM).to(device)#, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)


losses = []


def training(src,trg,eval_src,eval_trg):
    config = configs()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr = config.lr , weight_decay = config.decay)
    best_acc = 0
   
    for epoch in range(config.epoch):
        break
        model.train()
        total_loss = 0
       # print('\nDATA',data)
        for index, texts in tqdm(enumerate(src),total = configs.LTrain,leave=True,position = 0):
    
            
            
            output,loss = model(texts,trg)
            exit()
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



    validate(eval_src,eval_trg,criterion,best_acc)

def validate(eval_data,eval_trg,criterion,best_acc):
    #model.eval()
    total_loss = 0
    epoch_loss = 0
    mean_acc = 0
    total = 0
    correct = 0
    bleu_score = 0
    with torch.no_grad():
        for index, texts in tqdm(enumerate(eval_data),total=configs.LTest,position=0,leave=True):
            
           
            output,loss = model(texts,eval_trg)
            total_loss +=loss#.item()
            '''
            output_dim = output.shape[-1]
            output = output[0:].view(-1,output_dim)
            #trg = trg.squeeze(1)

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
            '''
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






print("5")
training(src_batch,trg_batch,test_src_batch,test_trg_batch)



(data_train10, trg_train10),(data_test10,trg_test10), vec10 = load_src_data(10)
(data_train50, trg_train50),(data_test50,trg_test50), vec50= load_src_data(50)

test_src_batch10,test_trg_batch10 = batching(data_test10,configs.batch_size),batching(trg_test10,configs.batch_size)
test_src_batch50,test_trg_batch50 = batching(data_test50,configs.batch_size),batching(trg_test50,configs.batch_size)

src_batch10,trg_batch10 = batching(data_train10,configs.batch_size),batching(trg_train10,configs.batch_size)
src_batch50,trg_batch50 = batching(data_train50,configs.batch_size),batching(trg_train50,configs.batch_size)

print("10-------------------------")
training(src_batch10,trg_batch10,test_src_batch10,test_trg_batch10)
print("50-------------------------")
training(src_batch50,trg_batch50,test_src_batch50,test_trg_batch50)

