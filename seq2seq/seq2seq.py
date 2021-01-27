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
    batch_size = 1
    lr = 2e-4
    decay = 0.9
    LTrain = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
      
        
    def forward(self,src,hidden,cell):
        
        #embedded = self.dropout(self.embedding(src))
        output = self.embedding(src)
        #output = output.unsqueeze(0)
        outputs, (hidden, cell) = self.rnn(output,(hidden,cell))
  
        return outputs,hidden, cell

    def initHidden(self):
        return torch.zeros(1, 1, self.hid_dim, device=device)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
     

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
        hidden = self.encoder.initHidden()
        cell = self.encoder.initHidden()
        #initialize a list of output, matching input dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        output,hidden,cell = self.encoder(src,hidden,cell)
        print(output)
        sys.exit()
        #<sos> first in 
        input = trg[0,:]
        #print(input)
        #sys.exit()
        #initialize <eos> of encoder, <sos> of decoder
        outputs[0,0,0] = 1
        
        outputs[6,0,1] = 1
        for t in range(1, trg_len-1):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            best = output.argmax(1)
            print(input)
            #0 for use real output, 1 for use best model output 
            input = trg[t] if random.random() > teacher_forcing_ratio else best
            #sys.exit()
            input = output
            '''
            if input[0] == 1:
                break
            '''
            #sys.exit()
        '''
        print("\n")
        print(t)
        print(src,outputs.shape)
        print(outputs.argmax(2))
        '''
        return outputs



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
 

'''
def load_trg_data(vec):
    data_file = 'target.txt'
    f = open(data_file,'r').readlines()
    tokenization = tokenize(f)
    decode, vec = char2vec(tokenization,vec)
    return decode[:5],decode[600:800],decode[800:]#, vec

test_train, test_val, test_test = load_trg_data(vec)
print(test_train)
sys.exit()
'''


INPUT_DIM = len(vec)
OUTPUT_DIM = len(vec)
ENC_EMB_DIM = 100
DEC_EMB_DIM = 100
HID_DIM = 100
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
    optimizer = optim.SGD(model.parameters(),lr = config.lr , weight_decay = config.decay)
    best_acc = 0
    for _ in range(config.epoch):
        model.train()
       # print('\nDATA',data)
        for _, (texts,trg) in tqdm(enumerate(data),total = configs.LTrain,leave=True):
        #for texts in data:
            '''
            print('\nsrc',texts)
            print('trg',trg)
            
            #sys.exit()
            '''
            texts = texts.unsqueeze(1)
            trg = trg.unsqueeze(1)
            #print(texts.shape)
            optimizer.zero_grad()
            output = model(texts,trg).to(device)
            
            output_dim = output.shape[-1]
            output = output[0:].view(-1,output_dim)
            #print(output.argmax(dim=1))
            texts = texts.squeeze(1)
            trg = trg.squeeze(1)
            print(('pred,',output.argmax(1)),('trg',trg))

            loss = criterion(output,trg)
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

#print(torch.transpose(cat_training,0,1))
#sys.exit()
training(torch.transpose(cat_training,0,1),torch.transpose(cat_test,0,1))


'''
plt.plot(losses)
plt.title('Seq2Seq Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
'''