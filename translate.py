import argparse

import dill
import torch
from torch.autograd import Variable

from train_utils import greedy_decode
from models import EncoderDecoder, Encoder, EncoderLayer, MultiHeadedAttention
from models import PositionwiseFeedForward, SublayerConnection, LayerNorm
from models import Decoder, DecoderLayer, Embeddings, PositionalEncoding
from models import Generator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--src-file-path', type=str)
    parser.add_argument('--result-file-path', type=str)
    return parser.parse_args()


def main(args):

    with open(args.model_path, 'rb') as f:
        model, SRC, TGT = dill.load(f)
    src_sents = open(args.src_file_path).readlines()
    result_sents = []

    model.eval()

    for sent in src_sents:
        sent = sent.split()
        src = torch.LongTensor([[SRC.vocab.stoi[w] for w in sent]])
        src = Variable(src)
        src_mask = (src != SRC.vocab.stoi['blank']).unsqueeze(-2)
        out = greedy_decode(model, src, src_mask,
                            max_len=60, start_symbol=TGT.vocab.stoi['<s>'])

        trans = '<s> '
        for i in range(1, out.size(1)):
            sym = TGT.vocab.itos[out[0, i]]
            if sym == '</s>':
                break
            trans += sym + ' '
        trans = trans.replace('<s> ', '')
        result_sents.append(trans)

    open(args.result_file_path, 'w').write('\n'.join(result_sents) + '\n')


if __name__ == '__main__':
    args = get_args()
    main(args)
