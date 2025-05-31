import re
import collections


def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs


def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out


vocab = {'l o w </w>': 5, 'l o w e r </w>': 2,
         'n e w e s t </w>': 6, 'w i d e s t </w>': 3}
num_merges = 10

for i in range(num_merges):
    pairs = get_stats(vocab)
    print(pairs)
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
print(vocab)
print(best)


# # Define the pair function
# def pairss(vocab):
#     pair_dic = {}

#     for word, freq in vocab.items():
#         tokens=word.split()
#         for i in range(len(tokens)-1):
#             if (tokens[i], tokens[i+1]) not in pair_dic.keys():
#                 pair_dic[(tokens[i], tokens[i+1])]=freq
#             else:
#                 pair_dic[(tokens[i], tokens[i+1])]+=freq
#     return pair_dic


# def merge(pair, v_in):
#     v_out = {}
#     for key in v_in:
#         chars = key.split()
#         new=''
#         i=0
#         while i<len(chars)-1:
#             if (chars[i], chars[i+1])==pair:
#                 new=new+' '+chars[i]+chars[i+1]
#                 i+=2
#             else:
#                 new=new+' '+chars[i]
#                 if i+1==len(chars)-1:
#                     new=new+chars[i+1]
#                 i+=1
#         v_out[new]=v_in[key]
#     return v_out
    
# # if __name__=='__main__':
# for i in range(num_merges):
#     pairs = pairss(vocab)
#     print(pairs)
#     best = max(pairs, key=pairs.get)
#     vocab = merge(best, vocab)
# print(vocab)
# print(best)
