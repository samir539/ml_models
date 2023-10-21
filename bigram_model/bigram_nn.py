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
    def __init__(self, depth,alphabet_len=27):
        super.__init__()
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
    for n in names_data[:2]:
        characters = ["."] + list(n) + ["."]
        for i,j in zip(characters,characters[1:]):
            L1.append(char_to_int[i])
            L2.append(char_to_int[j])
    L1,L2 = torch.tensor(L1), torch.tensor(L2)

    #one hot encode 
    train_data = F.one_hot(L1,num_classes=27)
    test_data = F.one_hot(L2, num_classes=27)
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
    for i in range(steps):
        out_probs = net(data)
        loss = -out_probs[torch.arange(num),L2].log().mean()
        print(loss.item())

        loss.backward()
        with torch.no_grad():
            for param in net.parameters():
                param -= lr * param.grad

        #step 
    




def make_names():
    """
    function to make names (run model in inference)
    """


with open("./names.txt", mode='r', encoding="utf-8") as f:
    names_data = f.read()


#put names into list
names_data = names_data.splitlines()

#char int conversion 
characters = list(sorted(set(''.join(names_data))))
characters.insert(0,'.')
char_to_int = {char_val:idx for (idx,char_val) in enumerate(characters)}
int_to_char = {idx:char_val for (idx, char_val) in enumerate(characters)}

get_data(names_data)