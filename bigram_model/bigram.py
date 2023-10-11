import torch
import torch.nn as nn


with open("./names.txt", mode='r', encoding="utf-8") as f:
    names_data = f.read()


#put names into list
names_data = names_data.splitlines()

##bigram model 

#generate bigrams

def generate_bigram(names_data):
    """
    function to generate bigrams for input data
    :param names_data: input data
    :return bigrams: a list of bigrams for the given input data
    """
    #dictionary of bigrams
    bigram_dict = {}
    for i in names_data:
        characters = ["<S>"] + list(i) + ["<E>"]
        for j,p in zip(characters,characters[1:]):
            bigram = (j,p)
            bigram_dict[bigram] = bigram_dict.get(bigram,0) + 1
    return bigram_dict


b = generate_bigram(names_data[:2])
print(b)     

#store bigram information in a second order tensor
for i in b:
    print(i)

