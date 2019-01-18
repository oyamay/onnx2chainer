import chainer.link
import chainer.functions as F
import numpy as np

from onnx2chainer.util import getNodeAttribute, getOneSidePads, getAutoPadWidth, printONNX2ChainerWarning

#

class OGlobalAveragePool(chainer.Chain):
    def __init__(self):
        super(OGlobalAveragePool, self).__init__()

    def forward(self, x):
        return F.average_pooling_nd(x, x.shape[2:])

class OMaxAveragePool(chainer.Chain):
    def __init__(self, average, ksize, stride, pad, auto_pad, return_indices):
        super(OMaxAveragePool, self).__init__()
        self.average = average
        self.ksize = ksize
        self.stride = stride
        self.pad = pad
        self.auto_pad = auto_pad
        self.return_indices = return_indices

    def forward(self, x):
        # TODO: merge with OConv
        if self.auto_pad:
            self.pad = [None]*(x.ndim-2)
            for i in range(x.ndim-2):
                w = x.shape[2+i]
                assert self.ksize[i] == self.stride[i]
                assert (w%self.ksize[i]) == 0
                p = getAutoPadWidth(w,
                                    int(w / self.ksize[i]),
                                    self.ksize[i],
                                    self.stride[i],
                                    1,
                                    self.auto_pad)
                assert p[0] == p[1]
                self.pad[i] = p[0]

            self.auto_pad = None

        n = len(self.ksize)
        assert n == x.ndim-2
        f = getattr(F, "{}_pooling_{}d".format("average" if self.average else "max",
                                               n if n <= 3 else "n"))

        args = {"ksize": self.ksize,
                "stride": self.stride,
                "pad": self.pad}

        if not self.average:
            args["cover_all"] = False
            args["return_indices"] = self.return_indices

        return f(x, **args)

class OMaxAveragePoolWithExplicitPadding(OMaxAveragePool):
    def __init__(self, average, ksize, stride, dpad, return_indices):
        super(OMaxAveragePoolWithExplicitPadding, self).__init__(average, ksize, stride, [0]*len(stride), None, return_indices)

        dpad = list(dpad)
        ndim = len(stride)
        assert len(dpad) == ndim*2
        starts = dpad[:ndim]
        ends = dpad[ndim:]
        self.dpad = np.zeros([2+ndim, 2], dtype=np.int)
        self.dpad[2:,0] = starts
        self.dpad[2:,1] = ends

    def forward(self, x):
        h = F.pad(x, self.dpad, mode="constant")
        return super(OMaxAveragePoolWithExplicitPadding, self).forward(h)

#

def parse_GlobalAveragePool(op, inits):
    return OGlobalAveragePool()

def parse_MaxPool(op, inits):
    return parse_MaxAveragePool(op, inits, False)

def parse_AveragePool(op, inits):
    return parse_MaxAveragePool(op, inits, True)

def parse_MaxAveragePool(op, inits, average):
    assert getNodeAttribute(op, "storage_order", 0) == 0

    auto_pad = getNodeAttribute(op, "auto_pad", "NOTSET")
    return_indices = len(op.output) == 2
    ksize  = getNodeAttribute(op, "kernel_shape")
    stride = getNodeAttribute(op, "strides")
    doubleSidedPads = getNodeAttribute(op, "pads", [0]*len(stride))
    try:
        pad = getOneSidePads(doubleSidedPads, assertEvens=True)
    except:
        printONNX2ChainerWarning("Padding widths of at least one dimension is not the same ({}).".format(doubleSidedPads) \
                                 + " This will incurs extra memory copy since Chainer does not support it.")
        return OMaxAveragePoolWithExplicitPadding(average, ksize, stride, doubleSidedPads, return_indices)

    if len(set(pad)) == 1:
        pad = pad[0]

    return OMaxAveragePool(average,
                           ksize, stride, pad,
                           auto_pad if auto_pad != "NOTSET" else None,
                           return_indices)
