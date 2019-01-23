import chainer
import chainer.functions as F

#

class OSub(chainer.Chain):
    def __init__(self):
        super(OSub, self).__init__()

    def forward(self, x1, x2):
        return x1-x2

class OMul(chainer.Chain):
    def __init__(self):
        super(OMul, self).__init__()

    def forward(self, x1, x2):
        return x1*x2

#

def parse_Add(op, inits):
    return F.add

def parse_Sum(op, inits):
    return F.add

def parse_Sub(op, inits):
    return OSub()

def parse_Mul(op, inits):
    return OMul()

def parse_Abs(op, inits):
    return F.absolute

def parse_ReduceSum(op, inits):
    return F.sum
