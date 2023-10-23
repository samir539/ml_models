import torch 
import torch.nn as nn
import torch.nn.functional as F


## get data 
with open("./utils/tiny_shakespear.txt", mode='r', encoding='utf8') as f:
    text_data = f.read()


##characters 
characters = sorted(list(set(text_data)))

## make mapping from characters to integers (tokeniser)
char_to_int = {ic:ix for ix,ic in enumerate(characters)}
int_to_char = {ix:ic for ix,ic in enumerate(characters)}

#extend to multiple characters
encode = lambda string_input:[char_to_int[x] for x in string_input]
decode = lambda int_input: ''.join([int_to_char[x] for x in int_input])


## tokenise entire tiny_shakespear text
tokenised_shakespear = encode(text_data)
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

x,y =make_batches(4,8,text_data)
print(x,"\n",y)





#bigram language model 
class bigram(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,vocab_size)
    
    def forward(self,x,y=None):
        logits = self.token_embedding_table(x)
        if y is None:
            loss = None
        else:
            logits = logits.view(logits.shape[0]*logits.shape[1],logits.shape[2])
            y = y.view(y.shape[0]*y.shape[1])
            loss = F.cross_entropy(logits,y)
        return logits,loss
    
    def generate(self,ix,tokens_to_generate):
        for _ in range(tokens_to_generate):
            #run forward 
            logits , loss = self.forward(ix)
            logits = logits[:,-1,:]
            prob_vec = F.softmax(logits,dim=1)
            ix_new = torch.multinomial(prob_vec,1,replacement=True)
            ix = torch.cat((ix,ix_new),dim=1)
        return ix
         



## train model 
def train_model(net, iters, learning_rate,block_size,text_data):
    """
    function to train the basic bigram model 

    """
    optimiser = torch.optim.AdamW(net.parameters(), lr = learning_rate)
    for _ in range(iters):
        xb, yb = make_batches(batch_size,block_size,text_data)

        #loss 
        logits, loss = net.forward(xb,yb)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()


#generate 
net = bigram(65)
train_model(net,10000,0.1,8,text_data)
context = torch.zeros((1,1), dtype=torch.long)
print(decode(net.generate(context,100)[0].tolist()))


