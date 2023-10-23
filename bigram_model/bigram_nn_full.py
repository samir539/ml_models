## full bigram neural net model

import torch 
import torch.nn as nn
import torch.nn.functional as F


## load in data 
with open("./names.txt", mode='r', encoding='utf8') as f:
    names_data = f.read()

#put names into list
names_data = names_data.splitlines()

#char int conversion 
characters = list(sorted(set(''.join(names_data))))
characters.insert(0,'.')
char_to_int = {char_val:idx for (idx,char_val) in enumerate(characters)}
int_to_char = {idx:char_val for (idx, char_val) in enumerate(characters)}

## build data set 
def make_dataset(names_data, context_length):
    """
    function to build the dataset 
    :param names_data: the data containing our names
    :param context_length: the context used
    :return context_tensor: the tensor containing the context
    :return target_tensor: the tensor containing the respective target for the corresponding context
    """
    context_tensor = []
    target_tensor = []

    for names in names_data:
        characters =  list(names) + ["."]
        context_window = [0] * context_length
        # context_tensor.append(context_window)
        for j in characters:
            context_window = context_window[1:] + [char_to_int[j]]
            context_tensor.append(context_window)
            target_tensor.append([char_to_int[j]])

    #convert to tensor 
    context_tensor = torch.tensor(context_tensor)
    target_tensor = torch.tensor(target_tensor).squeeze()


    return context_tensor, target_tensor


#give X we want to predict y (ie given a context of letter we want to predict the next char)

## make embeddings
#given a character we want a 2d vector to represent it 



## nn implementation 

class bigram_nn(nn.Module):
    def __init__(self, neurons, context_length = 3, embedding_space=2, alphabet_length=27):
        super().__init__()
        self.context_length = context_length
        self.embedding_space = embedding_space
        self.embeddings = torch.randn((27,embedding_space))
        self.linear1 = nn.Linear(embedding_space*context_length, neurons)
        self.linear2 = nn.Linear(neurons, alphabet_length)
        self.activation = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self,input):
        # print(input.shape)
        out = self.embeddings[input]
        ## need to change shape
        out = out.view(input.shape[0],self.context_length * self.embedding_space)
        out = self.linear1(out)
        out = self.activation(out)
        out = self.linear2(out)
        out = self.softmax(out)
        return out 



## training loop 
def training_loop(net, train_data,target_data, iters, lr):
    """
    function to implement a training loop
    :param net: an instance of the net object
    :param train_data: an instance of train data
    :param iters: number of training iters
    """
    for i in range(iters):
        net.zero_grad()

        #forward pass
        mb_index = torch.randint(0,train_data.shape[0],(32,))
        outputs = net.forward(train_data[mb_index])

        #get loss
        # print(outputs.shape, target_data.squeeze())
        loss = F.cross_entropy(outputs,target_data[mb_index])
        print("the loss on iteration {} is {}".format(i,loss))
        loss.backward()

        with torch.no_grad():
            for param in net.parameters():
                param += -lr*param.grad


def gen_names(num_names, net, context_length):
    """
    Function to use the trained bigram model to generate names 
    :param num_names: the number of names to generate 
    :param net: the train network
    """
    # loop per num names 
    #start with ...
    #pass to model 
    #loop while
    # pick from prob vector
    #break loop if choice is .
    for i in range(num_names):
        name = []
        ix = [0] * context_length
        # ix = torch.tensor(ix)
        # ix = ix.reshape((1,ix.shape[0]))
        # print("the shape of ix", ix.shape)
        # print(ix)
        while True:
            prob_vec = net.forward(torch.tensor(ix).reshape((1,len(ix))))
            # print("this is prob vec", prob_vec)
            idx = torch.multinomial(prob_vec,1,replacement=True).item()
            chr = int_to_char[idx]
            name.append(chr)
            ix = ix[1:] + [idx]
            if idx == 0: 
                break
        # print("hello",i)
        print(''.join(name))


        
net = bigram_nn(1000)
train_data, target_data = make_dataset(names_data,3)
training_loop(net,train_data,target_data,1000,0.1)
gen_names(13,net,3)


        