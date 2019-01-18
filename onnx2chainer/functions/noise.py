import chainer.link
import chainer.functions as F

from onnx2chainer.util import getNodeAttribute

#

class ODropout(chainer.Chain):
    def __init__(self, ratio, return_mask):
        super(ODropout, self).__init__()
        self.ratio = ratio
        self.return_mask = return_mask

    def forward(self, x):
        return F.dropout(x,
                         ratio=self.ratio,
                         return_mask=self.return_mask)

#

def parse_Dropout(op, inits):
    mask = len(op.output) == 2
    return ODropout(ratio=getNodeAttribute(op, "ratio", 0.5),
                    return_mask=mask)
