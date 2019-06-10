import chainer.link
import chainer.links as L
import chainer.functions as F
import numpy as np

from onnx2chainer.util import which, getNodeAttribute, getOneSidePads, getAutoPadWidth

#

class OConv(chainer.Chain):
    def __init__(self, stride, pad, auto_pad, dilate, groups, W, b=None):
        super(OConv, self).__init__()

        self.in_channels = W.shape[1] * groups
        self.out_channels = W.shape[0]
        self.ksize = W.shape[2:]
        self.stride = stride
        self.pad = pad
        self.auto_pad = auto_pad
        self.dilate = dilate
        self.groups = groups
        self.W = W
        self.b = b

        self.registered = False
        # self.l = None

    def forward(self, x):
        if not self.registered:
            self.registered = True

            if self.auto_pad:
                self.pad = [None]*(x.ndim-2)
                for i in range(x.ndim-2):
                    p = getAutoPadWidth(x.shape[2+i],
                                        x.shape[2+i],
                                        self.ksize[i],
                                        self.stride[i],
                                        self.dilate[i],
                                        self.auto_pad)
                    assert p[0] == p[1]
                    self.pad[i] = p[0]

            with self.init_scope():
                self.l = L.ConvolutionND(self.W.ndim-2,
                                         in_channels  = self.in_channels,
                                         out_channels = self.out_channels,
                                         ksize        = self.ksize,
                                         stride       = self.stride,
                                         pad          = self.pad,
                                         cover_all    = False,
                                         dilate       = self.dilate,
                                         groups       = self.groups,
                                         nobias       = (self.b is None),
                                         initialW     = self.W,
                                         initial_bias = self.b)

        self.l.to_gpu() # TODO: Do this only if GPU is enabled by a user
        return self.l(x)

class OGemm(chainer.Chain):
    def __init__(self, W, b=None):
        super(OGemm, self).__init__()

        in_size = W.shape[1]
        out_size = W.shape[0]

        with self.init_scope():
            self.l = L.Linear(in_size      = in_size,
                              out_size     = out_size,
                              nobias       = (b is None),
                              initialW     = W,
                              initial_bias = b)

    def forward(self, x):
        return self.l(x)

def parse_Conv(op, inits):
    stride = getNodeAttribute(op, "strides")

    auto_pad = getNodeAttribute(op, "auto_pad", "NOTSET")
    pad = getOneSidePads(getNodeAttribute(op, "pads", [0]*len(stride)))
    if len(set(pad)) == 1:
        pad = pad[0]

    return OConv(stride,
                 pad,
                 auto_pad if auto_pad != "NOTSET" else None,
                 getNodeAttribute(op, "dilations", 1),
                 getNodeAttribute(op, "group", 1),
                 inits[0], inits[1] if len(inits) >= 2 else None)

def parse_Gemm(op, inits):
    assert getNodeAttribute(op, "alpha", 1) == 1
    assert getNodeAttribute(op, "beta",  1) == 1 or len(inits) == 1
    assert getNodeAttribute(op, "transA", 0) == 0
    assert getNodeAttribute(op, "transB", 0) == 1
    return OGemm(inits[0], inits[1] if len(inits) >= 2 else None)

def parse_MatMul(op, inits):
    return F.matmul
