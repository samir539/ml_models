## load in tiny shakespear
with open("tiny_shakespear.txt", 'r', encoding="utf-8") as f:
    text = f.read()


## analyse tiny shakespear
#number of unique characters
char_list = sorted(list(set(text))) #sorted list
char_num = len(char_list)

## make mapping from characters to integers
char_to_int = lambda char: char_list.index(char) #given a char we get an integer 
int_to_char = lambda int: char_list[int] #given an integer we get a char

#extend to multiple characters
string_to_ints = lambda string_input:[char_to_int(x) for x in string_input]
ints_to_string = lambda int_input: [int_to_char(x) for x in int_input]

print(string_to_ints("hello world"))
print(ints_to_string([46, 43, 50, 50, 53, 1, 61, 53, 56, 50, 42]))

# lambda string: for i in string 

