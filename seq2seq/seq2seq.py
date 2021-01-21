#import pytorch stuff
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
    epoch = 10
    batch_size = 32
    lr = 1e-5
    decay = 0.97

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        
        #self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        #embedded = self.dropout(self.embedding(src))
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.rnn(embedded)
        
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        #self.dropout = nn.Dropout(dropout)
        
       
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputx, hidden, cell):
        
        inputx = inputx.unsqueeze(0)

        #embedded = self.dropout(self.embedding(inputx))
        embedded = self.embedding(inputx)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
       
        pred = self.fc_out(output.squeeze(0))
        
        return pred, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0):
        
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = len(vec)#self.decoder.output_dim
        
        #initialize a list of output, matching input dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #encoder output to context vector
        hidden, cell = self.encoder(src)
        
        #<sos> first in 
        input = trg[0,:]
        
        #initialize <eos> of encoder, <sos> of decoder
        outputs[0,0,0] = 1
        outputs[6,0,1] = 1
        for t in range(1, trg_len-1):
        
            output, hidden, cell = self.decoder(input, hidden, cell)

            outputs[t] = output
            best = output.argmax(1) 
            
            #0 for use real output, 1 for use best model output 
            input = trg[t] if random.random() > teacher_forcing_ratio else best
            #print(input)
        '''
        print("\n")
        print(t)
        print(src,outputs.shape)
        print(outputs.argmax(2))
        '''
        return outputs



def char2vec(word_list):
    vec = {'\t':0,'\n':1}
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

def load_data():
    data_file = 'samples.txt'
    f = open(data_file,'r').readlines()
    tokenization = tokenize(f)
    encode, vec = char2vec(tokenization)
    return encode[:600],encode[600:800],encode[800:], vec

data_train, data_val, data_test, vec = load_data()

print(data_train.shape)

epoch = 20

INPUT_DIM = len(vec)
OUTPUT_DIM = len(vec)
ENC_EMB_DIM = 30
DEC_EMB_DIM = 30
HID_DIM = 16
N_LAYERS = 1
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1


enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)


losses = []


def training(data,eval_data):
    config = configs()
    #model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr = config.lr , weight_decay = config.decay)
    best_acc = 0
    for _ in range(epoch):
        model.train()
        for _, texts in tqdm(enumerate(data),total=600,position=0,leave=True):
            texts = texts.unsqueeze(1)
            #print(texts.shape)
            optimizer.zero_grad()
            output = model(texts,texts).to(device)
            
            output_dim = output.shape[-1]
            output = output[0:].view(-1,output_dim)
            #print(output.argmax(dim=1))
            texts = texts.squeeze(1)
            loss = criterion(output,texts)
            loss.backward()
            optimizer.step()
        print(loss)
        losses.append(loss)
        validate(eval_data,criterion,best_acc)




def validate(eval_data,criterion,best_acc):
    model.eval()
    epoch_loss = 0
    mean_acc = 0
    total = 0
    correct = 0
    with torch.no_grad():
    
        for _, texts in tqdm(enumerate(eval_data),total=200,position=0,leave=True):

            texts = texts.unsqueeze(1)

            output = model(texts,texts,0)
            output_dim = output.shape[-1]
            output = output[0:].view(-1,output_dim)
            texts = texts.squeeze(1)

            pred =  output.argmax(1)
            lossE = criterion(output,texts)

            total += texts.size(0)
            correct += (pred==texts).sum().item()
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

training(data_train,data_val)


plt.plot(losses)
plt.title('Seq2Seq Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()