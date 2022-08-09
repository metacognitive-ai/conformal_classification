import argparse
import sys
from typing import Any

from OpenSSL import SSL
import fasttext
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from utils import *
from torch.utils.data import DataLoader
from conformal import ConformalModel
import torch.backends.cudnn as cudnn
import random
import torchtext
from torchtext.vocab import FastText


parser = argparse.ArgumentParser(description='Conformalize Torch-FT Model')
#parser.add_argument('data', metavar='IMAGENETVALDIR', help='path to Imagenet Val')
#parser.add_argument('--batch_size', metavar='BSZ', help='batch size', default=128)
#parser.add_argument('--num_workers', metavar='NW', help='number of workers', default=0)
#parser.add_argument('--num_calib', metavar='NCALIB', help='number of calibration points', default=10000)
parser.add_argument('--seed', metavar='SEED', help='random seed', default=0)

if __name__ == "__main__":
    args = parser.parse_args()
    ### Fix randomness
    np.random.seed(seed=args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)


    #creating field
    TEXT = torchtext.legacy.data.Field(lower=True, include_lengths=True, fix_length=150)
    LABEL = torchtext.legacy.data.Field(sequential=True, use_vocab=False)

    INDEX = torchtext.legacy.data.Field(sequential=False)

    id_label = 'id'
    text_label = 'text'
    label = 'label'

    train_fields=[
        #(index, INDEX),
        (text_label, TEXT),
        (label, LABEL)]

    train = torchtext.legacy.data.TabularDataset(
        path='model_data/train.csv', format='csv', skip_header=True,
        fields=train_fields)

    test_fields=[
        #(id_label, INDEX),
        (text_label, TEXT),
        (label, LABEL)
    ]
    test = torchtext.legacy.data.TabularDataset(
        path='model_data/test.csv', format='csv', skip_header=True,
        fields=test_fields)


    '''
    vectors = FastText('tr')

    max_size = 30000
    TEXT.build_vocab(train, test, vectors=vectors, max_size=max_size)
    INDEX.build_vocab(test)
    ntokens = len(TEXT.vocab)
    print (ntokens)
    '''
    calib_loader= DataLoader(train, batch_size=16, shuffle=True)
    val_loader = DataLoader(test, batch_size=16, shuffle=True)

    '''
    #iterator for training Neural network models
    train_iterator = torchtext.legacy.data.BucketIterator(calib_loader, batch_size=32,
                                                          sort_key=lambda x: len(x.text),
                                                          sort_within_batch=True, repeat=False)
    test_iterator = torchtext.legacy.data.BucketIterator(val_loader, batch_size=32,
                                                         sort_key=lambda x: len(x.text),
                                                         sort_within_batch=True, train=False, repeat=False)
    '''

    # Initialize loaders
    cudnn.benchmark = True

    # Get the model
    model = fasttext.load_model("model_data/61d84d0d632e1d67f87de8dd_TrainingIntent.bin")
    model = torch.nn.DataParallel(model)
    model.eval()

    # optimize for 'size' or 'adaptiveness'
    lamda_criterion = 'size'
    # allow sets of size zero
    allow_zero_sets = False
    # use the randomized version of conformal
    randomized = True

    # Conformalize model
    model = ConformalModel(model, calib_loader, alpha=0.1, lamda=0, randomized=randomized, allow_zero_sets=allow_zero_sets)

    print("Model calibrated and conformalized! Now evaluate over remaining data.")
    validate(val_loader, model, print_bool=True)

    print("Complete!")
