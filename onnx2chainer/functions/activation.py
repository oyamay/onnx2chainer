import chainer.link
import chainer.functions as F
from chainer import link

from onnx2chainer.util import getNodeAttribute

#

class OSoftmax(chainer.Chain):
    def __init__(self, axis):
        super(OSoftmax, self).__init__()
        self.axis = axis

    def forward(self, x):
        return F.softmax(x, axis=self.axis)

#

def parse_Relu(op, inits):
    return F.relu

def parse_Softmax(op, inits):
    return OSoftmax(axis=getNodeAttribute(op, "axis", 1))
