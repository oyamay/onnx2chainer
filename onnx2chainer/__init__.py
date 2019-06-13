import onnx
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from functools import reduce
if chainer.cuda.available:
    import cupy as cp

import onnx2chainer.util
from onnx2chainer.util import which
from onnx2chainer import functions

def onnxToChainer(o, gpu=None, printTensorStack=False):
    graph = o.graph

    params = {}
    for i in graph.initializer:
        dtype = onnx2chainer.util.elemTypeToNumpy(i.data_type)
        if len(i.raw_data) > 0:
            d = np.frombuffer(i.raw_data, dtype=dtype)

        elif len(i.float_data) > 0:
            d = np.array(i.float_data, dtype=dtype)

        elif len(i.int64_data) > 0:
            d = np.array(i.int64_data, dtype=dtype)

        else:
            assert False

        params[i.name] = d.reshape(i.dims)

    class DecodedChain(chainer.Chain):
        def __init__(self, graph, params):
            super(DecodedChain, self).__init__()
            self.nodes = []
            for i, op in enumerate(graph.node):
                cop = onnxNodeToChainer(op, params)
                if gpu is not None and hasattr(cop, "to_gpu"):
                    cop.to_gpu()

                self.nodes.append((cop, op.input, op.output))
                with self.init_scope():
                    setattr(self, "node_{}".format(i), cop)

            self.paramNames = params.keys()

            nodeInputNames = set(reduce(lambda a,b: a+b,
                                        map(lambda x: list(x[1]), self.nodes)))
            nodeOutputNames = set(reduce(lambda a,b: a+b,
                                         map(lambda x: list(x[2]), self.nodes)))

            self.modelInputNames = nodeInputNames - nodeOutputNames - set(self.paramNames)

        def forward(self, *args):
            tensors = {}

            assert len(self.modelInputNames) == len(args)
            for name, arg in zip(list(self.modelInputNames), args):
                tensors[name] = arg

            for op, iNames, oNames in self.nodes:
                inputs = list(map(lambda x: tensors[x],
                                  filter(lambda y: not y in self.paramNames, iNames))) # Ignore params. from parse_*'s arguments

                if isinstance(inputs, self.xp.ndarray):
                    inputs = inputs[0]

                if printTensorStack:
                    print(op)
                    for k in tensors.keys():
                        print("{} {}".format(k, tensors[k].shape if tensors[k] is not None else None))

                    print("-- {} -> {} --".format(iNames, oNames))

                outputs = op(*inputs)
                if isinstance(outputs, chainer.Variable):
                    assert len(oNames) == 1
                    outputs = [outputs]

                for i, oName in enumerate(oNames):
                    tensors[oName] = outputs[i]

            return outputs

    return DecodedChain(graph, params)

def onnxNodeToChainer(op, ginits):
    opType = op.op_type

    inits = []
    for iName in op.input:
        if iName in ginits.keys():
            inits.append(ginits[iName])

    converterName = "parse_{}".format(opType)
    if not hasattr(functions, converterName):
        print(onnx2chainer.util.printONNX2ChainerWarning("op_type \"{}\" is not supported.".format(opType)))
        exit()

    return getattr(functions, converterName)(op, inits)
