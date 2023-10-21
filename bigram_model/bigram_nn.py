## bigram model with neural network
import torch
import torch.nn as nn
import torch.nn.functional as F

class bigram_net(nn.Module):
    """
    class to implement bigram language model with a nn 

    Methods:
        __init__: Initialise an instance of the object

    """
    def __init__(self,alphabet_len=27):
        super().__init__()
        self.layer = nn.Linear(alphabet_len,alphabet_len)

    def forward(self,input):
        """
        Method to carry out a forward pass of the nn
        :param input: the input to the neural network
        """
        #one hot encode the input 
        out = self.layer(input)
        sftmax = nn.Softmax(dim=1) #softmax
        out = sftmax(out)
        # print("this is ",out.sum(dim=0))
        return out
    

def get_data(names_data):
    """
    function to prepare data
    takes list of names and return L1,L2 containing the train and test data respectively 
    :param names_data: input names data  
    :return train_data: pyTorch tensor with one hot enoded values for the train instances
    :return test_data: pyTorch tensor with one hot enoded values for the test instances
    """
    L1, L2 = [],[]
    for n in names_data:
        characters = ["."] + list(n) + ["."]
        for i,j in zip(characters,characters[1:]):
            L1.append(char_to_int[i])
            L2.append(char_to_int[j])
    L1,L2 = torch.tensor(L1), torch.tensor(L2)

    #one hot encode 
    train_data = F.one_hot(L1,num_classes=27).float()
    test_data = F.one_hot(L2, num_classes=27).float()
    return L1,L2,train_data,test_data
            


def training(data,L2,net,steps,lr):
    """
    function to carry out training process of model 
    """
    # - Make a prediction (forward pass)
    # - Calculate loss
    # - Calculate backward gradients
    # - Optimise
    # training loop 
    num = L2.nelement()
    print(num)
    for i in range(steps):
        net.zero_grad()
        out_probs = net(data)
        # print("these are out probs", out_probs.sum(dim=1))
        loss = -out_probs[torch.arange(num),L2].log().mean()
        print(loss.item())

        loss.backward()
        with torch.no_grad():
            for param in net.parameters():
                param += -lr * param.grad

        #step 





def make_names(num_of_names,net):
    """
    function to make names (run model in inference)
    :params num_of_names: the number of names to generate
    :param net: the trained bigram nn
    """
    gen = torch.Generator().manual_seed(77)
    for i in range(num_of_names):
        # print("this is i",i)
        name = []
        idx = 0
        while True:
            input_data = F.one_hot(torch.tensor([idx]), num_classes=27).float()
            # print(input_data)
            # print("this is net input data",net(input_data))
            nn_prob = net(input_data)
            # print("this is the out", test)
            # print("-----------------")
            # print("this is test",torch.argmax(test))
            # idx = torch.argmax(test)
            unif = torch.ones((1,27))/27
            idx = torch.multinomial(nn_prob,1,replacement=True, generator=gen).item()
            # print("this is idx",idx)
            if idx == 0:
                # print("this happened")
                break
            chr = int_to_char[idx]
            name.append(chr)
        print(''.join(name))

            






with open("./names.txt", mode='r', encoding="utf-8") as f:
    names_data = f.read()


#put names into list
names_data = names_data.splitlines()

#char int conversion 
characters = list(sorted(set(''.join(names_data))))
characters.insert(0,'.')
char_to_int = {char_val:idx for (idx,char_val) in enumerate(characters)}
int_to_char = {idx:char_val for (idx, char_val) in enumerate(characters)}

L1,L2, train_data, test_data = get_data(names_data)

bigram_nn = bigram_net()

training(train_data,L2,bigram_nn,1000,lr=1)

make_names(50,bigram_nn)







