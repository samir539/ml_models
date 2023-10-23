import torch 
import torch.nn.functional as F
# letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 
#         'm', 'n', 'o', 'p', 'q','r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z','<S>','<E>']
# char_to_int = lambda char: letters.index(char) #given a char we get an integer 
# tuple_to_index = lambda tuple: [char_to_int(i) for i in tuple ]

# print(tuple_to_index(('a','<E>')))
# c = tuple_to_index(('a','<E>'))
# print(tuple(c))
# bigram_tensor = torch.zeros(28,28)
# # b = (0,0)
# bigram_tensor[c[0],c[1]] = 10
# print(bigram_tensor)
with open("./names.txt", mode='r', encoding="utf-8") as f:
    names_data = f.read()


#put names into list
names_data = names_data.splitlines()

# characters = list(sorted(set(''.join(names_data))))
# characters.insert(0,'.')
# print(enumerate(characters))
# for i,j in enumerate(characters):
#     print(i,j)

# char_to_int = {char_val:idx for (idx,char_val) in enumerate(characters)}
# int_to_char = {idx:char_val for (idx, char_val) in enumerate(characters)}
# print(char_to_int)
# print(int_to_char)
# # print(F.one_hot(torch.arange(0,5)%3))
# a = torch.tensor([0,0,0.2,0.8,0])
# print(torch.argmax(a).item())

# for i in range(10):
#     print(i)

C = torch.randn((27,2))
X = torch.randint(0,27,(32,3))
print("this is C",C)
print(C[X])
# print(C[[1,2]])