import torch
import torch.nn as nn
from matplotlib import pyplot as plt


with open("./names.txt", mode='r', encoding="utf-8") as f:
    names_data = f.read()


#put names into list
names_data = names_data.splitlines()

################
##bigram model## 
################

#generate bigrams

def generate_bigram(names_data):
    """
    function to generate bigrams for input data
    :param names_data: input data
    :return bigrams: a list of bigrams for the given input data
    """
    bigram_dict = {}
    for i in names_data:
        characters = ["."] + list(i) + ["."]
        for j,p in zip(characters,characters[1:]):
            bigram = (j,p)
            bigram_dict[bigram] = bigram_dict.get(bigram,0) + 1
    
    return bigram_dict

def generate_bigram_tensor(bigram_dict, visualise=False):
    """
    function to generate bigram tensor given a bigram dict
    :param bigrams_dict: dictionary containing the bigram , frequency pairs
    :param visualise (optional): option to visualise bigram tensor 
    :return bigram_tensor: 2D tensor expressing bigram frequency 

    """
    #converisons
    letters = ['.','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 
        'm', 'n', 'o', 'p', 'q','r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    char_to_int = lambda char: letters.index(char) #given a char we get an integer
    int_to_char = lambda int: letters[int] 
    tuple_to_index = lambda tuple: [char_to_int(i) for i in tuple ] #convert tuple to index 
    
    #init bigram_tensor
    bigram_tensor = torch.zeros((27,27), dtype=torch.int32)
    freq = list(sorted(bigram_dict.items(), key=lambda kv: -kv[1]))
    for i in freq:
        indx = tuple_to_index(i[0])
        bigram_tensor[indx[0],indx[1]] = i[1]

    #visualisation
    if visualise == True:
        plt.figsize=(64,64)
        plt.imshow(bigram_tensor, cmap="magma")
        for i in range(27):
            for j in range(27):
                bigramstr =  int_to_char(i) + int_to_char(j)
                plt.text(j,i, bigramstr, ha="center", va="bottom", color="grey",fontsize=5)
                plt.text(j,i,bigram_tensor[i,j].item(), ha="center", va="top", color="grey",fontsize=5)
        plt.axis('off')
        plt.show()
    return bigram_tensor



b = generate_bigram(names_data)
out_tensor = generate_bigram_tensor(b,False)

##multimodal distribution with broadcasting


## generate matrix of probablities
## sample from the probabilities
## write a loop to generate bigram names 

def prob_mat(bigram_tensor):
    """
    Function to generate a corresponding tensor containing the  probablilties for each bigram 
    :param bigram_tensor: the frequency tensor of each possible bigram  
    :return bigram_prob_tensor: the bigram probability tensor 
    """
    # bigram_tensor = bigram_tensor.float()
    sumation_col = bigram_tensor.sum(1,keepdim=True)
    bigram_prob_tensor = bigram_tensor/sumation_col
    return bigram_prob_tensor

# print(prob_mat(out_tensor))



def generate_names(bigram_prob_tensor, num_names):
    """
    Function to generate novel names given a probability tensor
    :param bigram_prob_tensor: the bigram probablity tensor
    :param num_names: the numbers of names to generate
    """
    #converisons
    letters = ['.','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 
        'm', 'n', 'o', 'p', 'q','r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    char_to_int = lambda char: letters.index(char) #given a char we get an integer
    int_to_char = lambda int: letters[int]
    gen = torch.Generator().manual_seed(77)
    for i in range(num_names):
        gen_name = []
        letter_val  = 0
        while True:
            char = int_to_char(letter_val)
            gen_name.append(char)
            letter_val =  torch.multinomial(bigram_prob_tensor[letter_val,:],1,replacement=True, generator=gen).item()
            if letter_val == 0:
                break
        print(''.join(gen_name))
    

prob_mat_out = prob_mat(out_tensor)
generate_names(prob_mat_out,50)




    #for num names
    # inner loop
    # pick next letter 
    # if end character stop 
    # print out name
# torch.Generator
