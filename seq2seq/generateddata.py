import random



LETTER = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.lower()
sample_size = 6000
MAX_LEN = 5
f = open('padding.txt','w')

def generate():
    
 
    for i in range(sample_size):
        sample = ''
        generated = sample.join(random.choices(LETTER,k=random.randint(2,5)))
        
        f.write(generated + '\n')
        # print(sample.join(random.choices(LETTER,k=MAX_LEN)))

    f.close()
    


generate()

