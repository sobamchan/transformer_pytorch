import spacy

import torch
import torch.nn as nn
from torchtext import data, datasets

from models import make_model
from train_utils import LabelSmoothing, MyIterator, batch_size_fn
from train_utils import NoamOpt, run_epoch, rebatch, MultiGPULossCompute


spacy_de = spacy.load('de')
spacy_en = spacy.load('en')


def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = '<blank>'
SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
TGT = data.Field(tokenize=tokenize_en,
                 init_token=BOS_WORD,
                 eos_token=EOS_WORD,
                 pad_token=BLANK_WORD)

MAX_LEN = 100
train, val, test = datasets.IWSLT.splits(
        exts=('.de', '.en'),
        fields=(SRC, TGT),
        filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
        len(vars(x)['trg']) <= MAX_LEN
        )
MIN_FREQ = 2
SRC.build_vocab(train.src, min_freq=MIN_FREQ)
TGT.build_vocab(train.trg, min_freq=MIN_FREQ)

devices = [0, 1, 2]
if True:
    pad_idx = TGT.vocab.stoi['<blank>']
    model = make_model(len(SRC.vocab), len(TGT.vocab), n=6)
    model.cuda()
    criterion = LabelSmoothing(size=len(TGT.vocab),
                               padding_idx=pad_idx,
                               smoothing=0.1)
    criterion.cuda()
    # BATCH_SIZE = 12000
    BATCH_SIZE = 120
    train_iter = MyIterator(train,
                            batch_size=BATCH_SIZE,
                            device=0,
                            repeat=False,
                            sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn,
                            train=True)
    valid_iter = MyIterator(val,
                            batch_size=BATCH_SIZE,
                            device=0,
                            repeat=False,
                            sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn,
                            train=False)
    model_par = nn.DataParallel(model, device_ids=devices)


if True:
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
                        torch.optim.Adam(model.parameters(), lr=0,
                        betas=(0.9, 0.98), eps=1e-9))
    for epoch in range(10):
        model_par.train()
        run_epoch((rebatch(pad_idx, b) for b in train_iter),
                  model_par,
                  MultiGPULossCompute(model.generator, criterion,
                  devices=devices, opt=model_opt))

        model_par.eval()
        loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter),
                         model_par,
                         MultiGPULossCompute(model.generator, criterion,
                         devices=devices, opt=None))
        print(loss)
