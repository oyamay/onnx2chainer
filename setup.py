from setuptools import setup

setup(name="onnx2chainer",
      version="0.1.0",
      description="A Python package for the ONNX->Chainer model conversion",
      url="https://github.com/oyamay/onnx2chainer",
      author="Yosuke Oyama",
      author_email="oyama.y.aa@m.titech.ac.jp",
      license="BSD",
      packages=["onnx2chainer"],
      install_requires=["chainer>=5.1.0",
                        "onnx>=1.3.0",
                        "numpy>=1.16.0",
                        "nose>=1.3.7"],
      test_suite="nose.collector",
      tests_require=["nose"],
      zip_safe=False)
