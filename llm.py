import torch 

with open('input.txt', 'r', encoding='utf-8') as f:
    input_data = f.read()

print(str(len(input_data)) + " characters in input data")

# Get unique characters in the input data
#['\n', ' ', '!', '$', '&', "'", ',', '-', '.', '3', ':', ';', '?', 'A',....]
chars=sorted(list(set(input_data)))

#print(str(len(chars)) + " unique characters in input data")
#print("".join(chars))

vocab_size=len(chars)

stoi={}
for i, ch in enumerate(chars):
    stoi[ch] = i

itos={}
for i, ch in enumerate(chars):
    itos[i] = ch

# string > int
def encode(s):
    return [stoi[c] for c in s]

# int > string
def decode(l):
    return "".join(itos[i] for i in l)

print(encode("hello world"))
print(decode(encode("hello world")))

data = torch.tensor(encode(input_data), dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[:1000])

n = int(0.9 * len(data))
training_data=data[:n]
validation_data=data[n:]

torch.manual_seed(1337)
block_size=8
batch_size=4

# first 9 chars in training set
print(training_data[:block_size + 1])

#get training data
x=training_data[:block_size]
#get next item in array for prediction
y=training_data[1:block_size+1]

# for t in range(block_size):
#     context = x[:t+1]
#     target=y[t]
#     print(f"when input is {context}, target is {target}")

def get_batch(split):
    data=training_data if split == "train" else validation_data
    #4x random position to grab chunk from 0-(1000-8)
    ix = torch.randint(len(data)-block_size, (batch_size,))
    # stack rows from data(randomnumber 1-4) 
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x,y

xb,yb = get_batch("train")
print("inputs")
print(xb.shape)
print(xb)
print("targets")
print(yb.shape)
print(yb)
print("------------------")

for b in range(batch_size):
    for t in range(block_size):
        context= xb[b,:t+1]
        target=yb[b,t]
        print(f"when input is {context.tolist()}, target is {target}")