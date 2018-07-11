import os
import argparse

import dill
from torchtext import data
from torchtext import datasets

import torch
import torch.nn as nn

from models import make_model
from train_utils import LabelSmoothing, MyIterator, batch_size_fn
from train_utils import NoamOpt, run_epoch, rebatch, MultiGPULossCompute


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str)
    parser.add_argument('--output-path', type=str)
    parser.add_argument('--epoch', type=int)
    return parser.parse_args()


def get_dataset(dpath):
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = '<blank>'
    EN = data.Field(pad_token=BLANK_WORD)
    JA = data.Field(init_token=BOS_WORD,
                    eos_token=EOS_WORD,
                    pad_token=BLANK_WORD)

    train = datasets.TranslationDataset(
            path=os.path.join(dpath, 'train'),
            exts=('.en', '.ja'),
            fields=(EN, JA))
    val = datasets.TranslationDataset(
            path=os.path.join(dpath, 'dev'),
            exts=('.en', '.ja'),
            fields=(EN, JA))

    MIN_FREQ = 2
    EN.build_vocab(train.src, min_freq=MIN_FREQ)
    JA.build_vocab(train.trg, min_freq=MIN_FREQ)
    return train, val, EN, JA


def sort_key_fn(x):
    return (len(x.src), len(x.trg))


def run():
    best_val_loss = 100

    args = get_args()
    train, val, EN, JA = get_dataset(args.data_path)

    devices = [0]

    pad_idx = EN.vocab.stoi['<blank>']
    model = make_model(len(EN.vocab), len(JA.vocab), n=6)
    model.cuda()
    criterion = LabelSmoothing(size=len(JA.vocab),
                               padding_idx=pad_idx,
                               smoothing=0.1)
    criterion.cuda()
    BATCH_SIZE = 540
    train_iter = MyIterator(train,
                            batch_size=BATCH_SIZE,
                            device=0,
                            repeat=False,
                            # sort_key=sort_key_fn,
                            sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn,
                            train=True)
    valid_iter = MyIterator(val,
                            batch_size=BATCH_SIZE,
                            device=0,
                            repeat=False,
                            # sort_key=sort_key_fn,
                            sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn,
                            train=False)
    model_par = nn.DataParallel(model, device_ids=devices)

    model_opt = NoamOpt(model.src_embed[0].d_model,
                        1,
                        2000,
                        torch.optim.Adam(model.parameters(),
                                         lr=0,
                                         betas=(0.9, 0.98),
                                         eps=1e-9))

    for epoch in range(args.epoch):
        model_par.train()
        run_epoch((rebatch(pad_idx, b) for b in train_iter),
                  model_par,
                  MultiGPULossCompute(model.generator,
                                      criterion,
                                      devices=devices,
                                      opt=model_opt))

        model_par.eval()
        loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter),
                         model_par,
                         MultiGPULossCompute(model.generator,
                                             criterion,
                                             devices=devices,
                                             opt=None))
        print(loss)
        if best_val_loss > loss:
            best_val_loss = loss
            model.cpu()
            # torch.save((model_par, EN, JA), args.output_path)
            with open(args.output_path, 'wb') as f:
                # dill.dump((model_par, EN, JA), f)
                dill.dump((model, EN, JA), f)
            model.cuda()


if __name__ == '__main__':
    run()
