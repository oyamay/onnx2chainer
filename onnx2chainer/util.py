import sys
import onnx
import onnx.shape_inference
import numpy as np
import pickle
from functools import reduce

def which(l, ary, defVal="_"):
    ret = list(filter(l, ary))
    if len(ret) != 1:
        assert defVal is not "_"
        return defVal

    return ret[0]

def getConvOutputWidth(wi, k, p, s, d=1):
    return int(1+(wi+2*p-((k-1)*d+1))/s)

def getConvInputWidth(wo, k, p, s, d=1):
    return (wo-1)*s-2*p+((k-1)*d+1)

def getAutoPadWidth(wi, wo, k, s, d, auto_pad):
    tp = (wo-1)*s-wi + (k-1)*d+1
    start_l = int(np.floor(tp/2.0))
    start_u = int(np.ceil(tp/2.0))
    if auto_pad == "VALID":
        assert start_l == 0 and start_u == 0

    start = start_u if auto_pad == "SAME_LOWER" else start_l
    return [start, tp-start]

def printONNX2ChainerWarning(s):
    sys.stderr.write("onnx2chainer warning: {}\n".format(s))

def getNodeAttribute(op, tag, default=None):
    a = list(filter(lambda x: x.name == tag, op.attribute))
    if len(a) != 1:
        assert len(a)== 0 and default is not None
        return default

    a = a[0]
    t = a.type
    if t == onnx.AttributeProto.INTS:
        return a.ints
    elif t == onnx.AttributeProto.FLOAT:
        return a.f
    elif t == onnx.AttributeProto.INT:
        return a.i
    elif t == onnx.AttributeProto.STRING:
        return a.s.decode("utf-8")

    if default is not None:
        return default

    assert False

def elemTypeToNumpy(t):
    if t == onnx.TensorProto.FLOAT:
        return np.float32
    elif t == onnx.TensorProto.UINT8:
        return np.uint8
    elif t == onnx.TensorProto.INT8:
        return np.int8
    elif t == onnx.TensorProto.UINT16:
        return np.uint16
    elif t == onnx.TensorProto.INT16:
        return np.int16
    elif t == onnx.TensorProto.INT32:
        return np.int32
    elif t == onnx.TensorProto.INT64:
        return np.int64
    elif t == onnx.TensorProto.STRING:
        return np.str
    elif t == onnx.TensorProto.BOOL:
        return np.bool
    assert False

def cloneNode(node):
    return pickle.loads(pickle.dumps(node))

def getTensorProducer(name, graph):
    for node in graph.node:
        if name in node.output:
            return node

    return None

def getTensorInitial(name, graph):
    for init in graph.initializer:
        if name == init.name:
            return np.frombuffer(init.raw_data, elemTypeToNumpy(init.data_type))

    return None

def getOneSidePads(pads, assertEvens=False):
    # [s1, s2, ..., e1, e2, ...] -> [s1, s2, ...]
    assert len(pads)%2 == 0
    count = int(len(pads)/2)

    begins = pads[:count]
    ends   = pads[count:]
    if not begins == ends:
        assert not assertEvens
        d = set(np.array(ends)-np.array(ends))
        assert d == set([0]) or d == set([0, 1]) # accept |p_end - p_begin| = 0 or 1
        printONNX2ChainerWarning("Padding widths of at least one dimension is not the same: {}".format(pads))

    return begins

def getDoubleSidePads(pads):
    return pads + pads

def getTensorShapes(model):
    inferredModel = onnx.shape_inference.infer_shapes(model)
    valueInfos = list(inferredModel.graph.value_info)
    valueInfos.extend(model.graph.input)
    valueInfos.extend(model.graph.output)

    ret = {}
    for vi in valueInfos:
        ret[vi.name] = list(map(lambda x: x.dim_value, vi.type.tensor_type.shape.dim))

    for init in inferredModel.graph.initializer:
        if not init.name in ret.keys():
            ret[init.name] = init.dims

    return ret

def getConvOutputWidthFromNode(w, node, conv=True):
    kernels   = which(lambda x: x.name == "kernel_shape", node.attribute).ints
    pads      = which(lambda x: x.name == "pads",         node.attribute).ints
    strides   = which(lambda x: x.name == "strides",      node.attribute).ints
    dilations = which(lambda x: x.name == "dilations",    node.attribute).ints if conv else [1]*len(kernels)
    assert len(kernels) == 2
    pads = getOneSidePads(pads)

    assert len(set(kernels))   == 1
    assert len(set(pads))      == 1
    assert len(set(strides))   == 1
    assert len(set(dilations)) == 1

    return getConvOutputWidth(w, kernels[0], pads[0], strides[0], dilations[0])

def createZeroTensor(name, dims):
    return onnx.helper.make_tensor(name=name,
                                   data_type=onnx.TensorProto.FLOAT,
                                   dims=dims,
                                   vals=np.zeros(reduce(lambda a,b: a*b, dims),
                                                 dtype=np.float32).reshape(*dims).tobytes(),
                                   raw=True)
