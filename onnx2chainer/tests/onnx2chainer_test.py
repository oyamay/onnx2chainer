import re
import os
import onnx
import onnx.numpy_helper
import numpy as np
import unittest
import subprocess
import chainer

import onnx2chainer

ONNX_MODEL_CACHE = "zoo_cache"

def loadPB(path):
    t = onnx.TensorProto()
    with open(path, "rb") as f:
        t.ParseFromString(f.read())

    return onnx.numpy_helper.to_array(t)

class TestOnnx2Chainer(unittest.TestCase):
    def _test(self, zooUrl, inputShape, outputShape=None):
        zooFileName = os.path.basename(zooUrl)
        zooPath = os.path.join(ONNX_MODEL_CACHE, zooFileName)
        if not os.path.exists(zooPath):
            subprocess.call(["wget", zooUrl, "--directory-prefix={}".format(ONNX_MODEL_CACHE)])

        zooFileNameWOExt, zooExt = re.compile("^([^.]+?)\.(.+)$").search(zooFileName).groups()
        if zooExt == "tar.gz":
            zooDecompressedPath = os.path.join(ONNX_MODEL_CACHE, zooFileNameWOExt)
            if not os.path.exists(zooDecompressedPath):
                subprocess.call(["tar", "-zxvf", zooPath, "-C", ONNX_MODEL_CACHE])

            zooPath = os.path.join(zooDecompressedPath, "model.onnx")

        oModel = onnx.load(zooPath)
        cModel = onnx2chainer.onnxToChainer(oModel)

        testSets = list(filter(lambda x: re.compile("^test_data_set_[0-9+]$").search(x), os.listdir(zooDecompressedPath)))
        testSets = sorted(testSets)

        for testSet in testSets:
            iPath = os.path.join(zooDecompressedPath, testSet, "input_0.pb")
            oPath = os.path.join(zooDecompressedPath, testSet, "output_0.pb")

            iTest = loadPB(iPath)
            assert iTest.shape == inputShape, (iTest.shape, inputShape)
            chainer.global_config.deterministic = True
            chainer.global_config.train = False
            oTest = cModel(iTest)[0]
            assert oTest.shape == outputShape, (oTest.shape, inputShape)

            oTestTruth = loadPB(oPath)

            oTestVec = oTest.data.reshape(-1)
            oTestTruthVec = oTestTruth.reshape(-1)

            diff = np.linalg.norm(oTestVec - oTestTruthVec) / np.linalg.norm(oTestTruthVec)
            print("Testing {:30} with {}: |y_chainer - y_true| / |y_true| = {:1.2e}".format(zooFileName, testSet, diff))

    def _test_imagenet(self, zooUrl):
        self._test(zooUrl,
                   (1, 3, 224, 224),
                   (1, 1000))

    def test_mnist(self):
        self._test("https://onnxzoo.blob.core.windows.net/models/opset_8/mnist/mnist.tar.gz",
                   (1, 1, 28, 28),
                   (1, 10))

    def test_bvlc_alexnet(self):
        self._test_imagenet("https://s3.amazonaws.com/download.onnx/models/opset_9/bvlc_alexnet.tar.gz")

    def test_bvlc_caffenet(self):
        self._test_imagenet("https://s3.amazonaws.com/download.onnx/models/opset_9/bvlc_reference_caffenet.tar.gz")

    def test_bvlc_googlenet(self):
        self._test_imagenet("https://s3.amazonaws.com/download.onnx/models/opset_9/bvlc_googlenet.tar.gz")

    def test_vgg19(self):
        self._test_imagenet("https://s3.amazonaws.com/download.onnx/models/opset_9/vgg19.tar.gz")

    def test_bvlc_resnet50(self):
        self._test_imagenet("https://s3.amazonaws.com/download.onnx/models/opset_9/resnet50.tar.gz")

if not os.path.exists(ONNX_MODEL_CACHE):
    os.makedirs(ONNX_MODEL_CACHE)

if __name__ == "__main__":
    unittest.main()
