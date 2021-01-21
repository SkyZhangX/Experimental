import random



LETTER = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.lower()
sample_size = 1000
MAX_LEN = 5
f = open('samples.txt','w')

def generate():
    
 
    for i in range(sample_size):
        sample = ''
        generated = sample.join(random.choices(LETTER,k=MAX_LEN))
        
        f.write(generated + '\n')
        # print(sample.join(random.choices(LETTER,k=MAX_LEN)))

    f.close()
    


generate()

