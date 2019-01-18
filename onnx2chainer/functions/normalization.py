import chainer.link
import chainer.functions as F
import chainer.links as L
from chainer import link

from onnx2chainer.util import getNodeAttribute

#

class OLRN(chainer.Chain):
    def __init__(self, n, k, alpha, beta):
        super(OLRN, self).__init__()
        self.n = n
        self.k = k
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        return F.local_response_normalization(x,
                                              n=self.n,
                                              k=self.k,
                                              alpha=self.alpha,
                                              beta=self.beta)

class OBatchNormalization(chainer.Chain):
    def __init__(self, scale, b, mean, var, decay, eps):
        super(OBatchNormalization, self).__init__()

        assert scale.ndim == 1 and len(set([scale.shape,
                                            b.shape,
                                            mean.shape,
                                            var.shape])) == 1
        size = scale.shape[0]

        with self.init_scope():
            self.l = L.BatchNormalization(size=size,
                                          decay=decay,
                                          eps=eps,
                                          use_gamma=True,
                                          use_beta=True,
                                          initial_gamma=scale,
                                          initial_beta=b,
                                          initial_avg_mean=mean,
                                          initial_avg_var=var)

    def forward(self, x):
        return self.l(x)

#

def parse_LRN(op, inits):
    size = getNodeAttribute(op, "size")
    return OLRN(n    =size,
                k    =getNodeAttribute(op, "bias", 1.0),
                alpha=getNodeAttribute(op, "alpha", 0.0001) / size,
                beta =getNodeAttribute(op, "beta", 0.75))

def parse_BatchNormalization(op, inits):
    assert getNodeAttribute(op, "spatial", 1) == 1
    assert len(op.output) == 1
    assert len(inits) == 4

    return OBatchNormalization(*inits,
                               decay = getNodeAttribute(op, "momentum", 0.9),
                               eps   = getNodeAttribute(op, "epsilon", 1e-5))
