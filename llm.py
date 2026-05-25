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
print(encode("hello world"))