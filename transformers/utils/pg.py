import torch
import random 

## load in tiny shakespear
with open("tiny_shakespear.txt", 'r', encoding="utf-8") as f:
    text = f.read()


## analyse tiny shakespear
#number of unique characters
char_list = sorted(list(set(text))) #sorted list
char_num = len(char_list)


## make mapping from characters to integers (tokeniser)
char_to_int = lambda char: char_list.index(char) #given a char we get an integer 
int_to_char = lambda int: char_list[int] #given an integer we get a char

#extend to multiple characters
encode = lambda string_input:[char_to_int(x) for x in string_input]
decode = lambda int_input: ''.join([int_to_char(x) for x in int_input])

print(encode("hello world"))
print(decode([46, 43, 50, 50, 53, 1, 61, 53, 56, 50, 42]))


## tokenise entire tiny_shakespear text
tokenised_shakespear = encode(text)
tokenised_shakespear = torch.tensor(tokenised_shakespear, dtype=torch.long)


##split into train and validation 
proportion = int(len(tokenised_shakespear)*0.9)
train_shakespear_tokens = tokenised_shakespear[:proportion]
test_shakespear_tokes = tokenised_shakespear[proportion:]

## make batches of context_length
context_length = 8
batch_size = 5

def make_batches(batch_size,context_length,dataset):
    """
    function to make batches of our training or validation data
    :param batch_size: the size of our batches
    :param context_length: the number of tokens each element of the batch
    :param dataset: the dataset to work with, either "train" or "test"
    :return x: randomly chosen batches of batch_size blocks of length context_length
    :return y: the target tokens for each block with our batch
    """
    #select dataset
    data = train_shakespear_tokens if dataset == "train" else test_shakespear_tokes
    rand_points = torch.randint(0,len(data)-context_length,(batch_size,)) #pick batchsize random points in our dataset 
    blocks_x = [data[x:x+context_length] for x in rand_points] #select blocks
    batch_x = torch.stack(blocks_x) #stack into single pyTorch tensor
    blocks_y = [data[x+1:x+context_length+1] for x in rand_points] #get the targets for all our blocks
    batch_y = torch.stack(blocks_y) #stack into pytorch tensor 
    return batch_x, batch_y






