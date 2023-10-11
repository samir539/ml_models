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
        characters = ["<S>"] + list(i) + ["<E>"]
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
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 
        'm', 'n', 'o', 'p', 'q','r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z','<S>','<E>']
    char_to_int = lambda char: letters.index(char) #given a char we get an integer
    int_to_char = lambda int: letters[int] 
    tuple_to_index = lambda tuple: [char_to_int(i) for i in tuple ] #convert tuple to index 
    #init bigram_tensor
    bigram_tensor = torch.zeros((28,28), dtype=torch.int32)
    freq = list(sorted(bigram_dict.items(), key=lambda kv: -kv[1]))
    for i in freq:
        indx = tuple_to_index(i[0])
        bigram_tensor[indx[0],indx[1]] = i[1]

    #visualisation
    if visualise == True:
        plt.figsize=(64,64)
        plt.imshow(bigram_tensor, cmap="magma")
        for i in range(28):
            for j in range(28):
                bigramstr =  int_to_char(i) + int_to_char(j)
                plt.text(j,i, bigramstr, ha="center", va="bottom", color="black",fontsize=5)
                plt.text(j,i,bigram_tensor[i,j].item(), ha="center", va="top", color="grey",fontsize=5)
        plt.axis('off')
        plt.show()
    return bigram_tensor



b = generate_bigram(names_data)
out_tensor = generate_bigram_tensor(b,True)


        





# b = generate_bigram(names_data[:4])
# print(b)
# print(generate_bigram_tensor(b))


