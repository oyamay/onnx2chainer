# onnx2chainer
This tool provides a way to convert [ONNX](https://github.com/onnx/onnx) models to [Chainer](https://github.com/chainer/chainer) models.

## Requirements
* Python >= 3.7.2
* [Chainer](https://github.com/chainer/chainer) >= 5.1.0
* [ONNX](https://github.com/onnx/onnx) >= 1.3.0
* [NumPy](http://www.numpy.org/) >= 1.16.0

The Python packages can be install via `pip3 install chainer onnx numpy`.

## How to Use
```python
import onnx
import onnx2chainer

model_onnx = onnx.load("path/to/onnx/model")
model_chainer = onnx2chainer.onnxToChainer(model_onnx)

x = a_xp_array
y = model_chainer(x)
```

See [`test/onnx2chainer_test.py`](test/onnx2chainer_test.py) for more details.

## How to Test
The test script [`test/onnx2chainer_test.py`](test/onnx2chainer_test.py)
1. downloads ONNX models as well as test datasets from [The ONNX Model Zoo](https://github.com/onnx/models) (might be O(GiB)),
2. generate a Chainer model from each of them, and
3. verify the generated models by comparing output of the models and grand-truth outputs, as follows.

```
Testing bvlc_alexnet.tar.gz            with test_data_set_0: |y_chainer - y_true| / |y_true| = 5.27e-07
Testing bvlc_reference_caffenet.tar.gz with test_data_set_0: |y_chainer - y_true| / |y_true| = 5.24e-08
Testing bvlc_googlenet.tar.gz          with test_data_set_0: |y_chainer - y_true| / |y_true| = 9.00e-07
Testing resnet50.tar.gz                with test_data_set_0: |y_chainer - y_true| / |y_true| = 3.70e-08
Testing mnist.tar.gz                   with test_data_set_0: |y_chainer - y_true| / |y_true| = 5.78e-04
Testing vgg19.tar.gz                   with test_data_set_0: |y_chainer - y_true| / |y_true| = 9.68e-07
```

Note that the error of [MNIST](https://github.com/onnx/models/tree/master/mnist) is relatively larger than others because the model does not have softmax.

## Supported ONNX Operations
* Add
* AveragePool
* BatchNormalization
* Concat
* Conv
* Dropout
* Gemm
* GlobalAveragePool
* LRN
* MatMul
* MaxPool
* Relu
* Reshape
* Softmax
* Split
* Sum
