import dill

import torch
from torch.autograd import Variable

from train_utils import greedy_decode
from models import EncoderDecoder, Encoder, EncoderLayer, MultiHeadedAttention
from models import PositionwiseFeedForward, SublayerConnection, LayerNorm
from models import Decoder, DecoderLayer, Embeddings, PositionalEncoding
from models import Generator


with open('en-ja-model.pt', 'rb') as f:
    model, SRC, TGT = dill.load(f)
sent = "how are you ?".split()
model.eval()
src = torch.LongTensor([[SRC.vocab.stoi[w] for w in sent]])
src = Variable(src)
src_mask = (src != SRC.vocab.stoi['<blank>']).unsqueeze(-2)
out = greedy_decode(model, src, src_mask,
                    max_len=60, start_symbol=TGT.vocab.stoi['<s>'])

trans = '<s> '
for i in range(1, out.size(1)):
    sym = TGT.vocab.itos[out[0, i]]
    if sym == '</s>':
        break
    trans += sym + ' '
print(trans)
