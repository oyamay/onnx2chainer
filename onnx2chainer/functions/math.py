import chainer.functions as F

def parse_Add(op, inits):
    return F.add

def parse_Sum(op, inits):
    return F.add
