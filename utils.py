import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import pathlib
import os
import pickle
from tqdm import tqdm
import pdb


# def sort_sum(scores):
#     # Dataloader sort sum from the original code
#     I = np.argsort(scores,axis=1)[:,::-1]
#     ordered = np.sort(scores,axis=1)[:,::-1]
#     cumsum = np.cumsum(ordered,axis=1)
#     return I, ordered, cumsum

def sort_sum(scores):
    # TODO: old code had axis=1 but the output of fasttext is 1-D only
    I = np.argsort(scores)
    ordered = np.sort(scores)
    cumsum = np.cumsum(ordered) 
    return I, ordered, cumsum

def sort_sum_dataloader(scores):
    # Dataloader sort sum from the original code
    I = np.argsort(scores,axis=1)[:,::-1]
    ordered = np.sort(scores,axis=1)[:,::-1]
    cumsum = np.cumsum(ordered,axis=1)
    return I, ordered, cumsum

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def validate_fasttext(helper, print_bool):
    batch_time = AverageMeter('batch_time')
    top1 = AverageMeter('top1')
    top5 = AverageMeter('top5')
    coverage = AverageMeter('RAPS coverage')
    size = AverageMeter('RAPS size')
    # switch to evaluate mode
    helper.cpmodel.eval()
    end = time.time()
    N = 0

    lbls = helper.datasets['test']['label']
    data = helper.datasets['test']['data']
    # for i, (x, target) in enumerate(val_loader):
    i = 0
    while i + helper.batch_size <= len(lbls):
        target = torch.from_numpy(np.array(lbls[i:(i+helper.batch_size)]))
        # compute output
        # This is conformal model, and it calls its own forward function
        output, S = helper.cpmodel(data[i:(i+helper.batch_size)])
        # measure accuracy and record loss
        prec1, prec5 = accuracy_fasttext(output, target, topk=(1, 5))
        cvg, sz = coverage_size_fasttext(S, target)

        # Update meters
        top1.update(prec1.item()/100.0, n=helper.batch_size)
        top5.update(prec5.item()/100.0, n=helper.batch_size)
        coverage.update(cvg, n=helper.batch_size)
        size.update(sz, n=helper.batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        N = N + helper.batch_size
        if print_bool:
            print(f'\rN: {N} | Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) | Cvg@1: {top1.val:.3f} ({top1.avg:.3f}) | Cvg@5: {top5.val:.3f} ({top5.avg:.3f}) | Cvg@RAPS: {coverage.val:.3f} ({coverage.avg:.3f}) | Size@RAPS: {size.val:.3f} ({size.avg:.3f})', end='')
        i = i + helper.batch_size
    if print_bool:
        print('') #Endline

    return top1.avg, top5.avg, coverage.avg, size.avg

def validate(val_loader, model, print_bool):
    with torch.no_grad():
        batch_time = AverageMeter('batch_time')
        top1 = AverageMeter('top1')
        top5 = AverageMeter('top5')
        coverage = AverageMeter('RAPS coverage')
        size = AverageMeter('RAPS size')
        # switch to evaluate mode
        model.eval()
        end = time.time()
        N = 0
        for i, (x, target) in enumerate(val_loader):
            target = target.cuda()
            # compute output
            output, S = model(x.cuda())
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            cvg, sz = coverage_size(S, target)

            # Update meters
            top1.update(prec1.item()/100.0, n=x.shape[0])
            top5.update(prec5.item()/100.0, n=x.shape[0])
            coverage.update(cvg, n=x.shape[0])
            size.update(sz, n=x.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            N = N + x.shape[0]
            if print_bool:
                print(f'\rN: {N} | Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) | Cvg@1: {top1.val:.3f} ({top1.avg:.3f}) | Cvg@5: {top5.val:.3f} ({top5.avg:.3f}) | Cvg@RAPS: {coverage.val:.3f} ({coverage.avg:.3f}) | Size@RAPS: {size.val:.3f} ({size.avg:.3f})', end='')
    if print_bool:
        print('') #Endline

    return top1.avg, top5.avg, coverage.avg, size.avg 

def coverage_size(S,targets):
    covered = 0
    size = 0
    for i in range(targets.shape[0]):
        if (targets[i].item() in S[i]):
            covered += 1
        size = size + S[i].shape[0]
    return float(covered)/targets.shape[0], size/targets.shape[0]

def coverage_size_fasttext(S,targets):
    covered = 0
    size = 0
    # TODO: assuming that S is 1-D. Adjust for multi D examples, e.g., resnet
    for i in range(targets.shape[0]):
        if (targets[i] in S):
            covered += 1
        size = size + len(S)  # TODO: previously S.shape[0], but S is a list object?
    return float(covered)/targets.shape[0], size/targets.shape[0]

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t().cuda()
    target = target.cuda()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def accuracy_fasttext(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    
    _, pred = output.topk(maxk)  # topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def data2tensor(data):
    imgs = torch.cat([x[0].unsqueeze(0) for x in data], dim=0).cuda()
    targets = torch.cat([torch.Tensor([int(x[1])]) for x in data], dim=0).long()
    return imgs, targets

def split2ImageFolder(path, transform, n1, n2):
    dataset = torchvision.datasets.ImageFolder(path, transform)
    data1, data2 = torch.utils.data.random_split(dataset, [n1, len(dataset)-n1])
    data2, _ = torch.utils.data.random_split(data2, [n2, len(dataset)-n1-n2])
    return data1, data2

def split2(dataset, n1, n2):
    data1, temp = torch.utils.data.random_split(dataset, [n1, dataset.tensors[0].shape[0]-n1])
    data2, _ = torch.utils.data.random_split(temp, [n2, dataset.tensors[0].shape[0]-n1-n2])
    return data1, data2

def get_model(modelname):
    if modelname == 'ResNet18':
        model = torchvision.models.resnet18(pretrained=True, progress=True)

    elif modelname == 'ResNet50':
        model = torchvision.models.resnet50(pretrained=True, progress=True)

    elif modelname == 'ResNet101':
        model = torchvision.models.resnet101(pretrained=True, progress=True)

    elif modelname == 'ResNet152':
        model = torchvision.models.resnet152(pretrained=True, progress=True)

    elif modelname == 'ResNeXt101':
        model = torchvision.models.resnext101_32x8d(pretrained=True, progress=True)

    elif modelname == 'VGG16':
        model = torchvision.models.vgg16(pretrained=True, progress=True)

    elif modelname == 'ShuffleNet':
        model = torchvision.models.shufflenet_v2_x1_0(pretrained=True, progress=True)

    elif modelname == 'Inception':
        model = torchvision.models.inception_v3(pretrained=True, progress=True)

    elif modelname == 'DenseNet161':
        model = torchvision.models.densenet161(pretrained=True, progress=True)

    else:
        raise NotImplementedError

    model.eval()
    model = torch.nn.DataParallel(model).cuda()

    return model

# Computes logits and targets from a model and loader
def get_logits_targets(helper):
    logits = torch.zeros((len(helper.datasets['train']['data']), helper.num_classes))
    labels = torch.zeros((len(helper.datasets['train']['data'])))
    i = 0
    print(f'Computing logits for model (only happens once).')

    lbls = helper.datasets['train']['label']
    data = helper.datasets['train']['data']

    
    # Iterate according to the batch size
    i = 0
    while i + helper.batch_size <= len(lbls):
        lgts = helper.get_logits(data[i:(i+helper.batch_size)])
        logits[i:(i+helper.batch_size), :] = torch.from_numpy(np.array(lgts))
        labels[i:(i+helper.batch_size)] = torch.from_numpy(np.array(lbls[i:(i+helper.batch_size)]))
        i = i + helper.batch_size

    # Construct the dataset
    dataset_logits = torch.utils.data.TensorDataset(logits, labels.long())
    return dataset_logits

