# vision-ai
We will use different small Offline AI models to perform some tasks like object detection etc. on Computer Vision like Images, videos etc. We will also try to develop the apps for mobile, where it is possible. We will mostly leverage Python, OpenCV, Onnxruntime, Numpy etc.

### Must include
```bash
sudo apt install -y ant autoconf automake ccache cmake g++ gcc lbzip2 libffi-dev libltdl-dev libtool libssl-dev make openjdk-17-jdk patch patchelf pkg-config ninja-build libbz2-dev

# python part 3.11
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-dev python3.11-venv

sudo ln -sf /usr/bin/python3.11 /usr/bin/python3
sudo ln -sf /usr/bin/python3.11 /usr/bin/python
sudo ln -sf /usr/bin/python3.11-config /usr/bin/python3-config
```

### Create patches

1. `Python::NumPy` in `onnxruntime_python.cmake`
```bash
diff -u onnxruntime_python.cmake onnxruntime_python_patch.cmake > onnx_numpy.patch
```
