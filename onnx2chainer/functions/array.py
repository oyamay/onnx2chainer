import chainer.link
import chainer.functions as F
import numpy as np

from onnx2chainer.util import getNodeAttribute

#

class OConcat(chainer.Chain):
    def __init__(self, axis):
        super(OConcat, self).__init__()
        self.axis = axis

    def forward(self, *xs):
        return F.concat(xs, axis=self.axis)

class OSplit(chainer.Chain):
    def __init__(self, axis, split):
        super(OSplit, self).__init__()
        self.axis = axis
        self.split = split

    def forward(self, x):
        return F.split_axis(x,
                            self.split,
                            axis=self.axis)

class OReshape(chainer.Chain):
    def __init__(self, shapes):
        super(OReshape, self).__init__()
        self.shapes = shapes

    def forward(self, x=None):
        inputs = self.shapes if x is None else [x] + self.shapes
        assert len(inputs) == 2

        return F.reshape(*inputs)

#

def parse_Concat(op, inits):
    return OConcat(axis=getNodeAttribute(op, "axis"))

def parse_Split(op, inits):
    axis = getNodeAttribute(op, "axis")
    split = getNodeAttribute(op, "split")
    if not isinstance(split, int):
        # [l1, l2, ...] -> [l1, l1+l2, ...]
        split = list(map(lambda x: np.sum(split[:x]).astype(np.int), range(len(split))))[1:]

    return OSplit(axis=axis,
                  split=split)

def parse_Reshape(op, inits):
    return OReshape(inits)
