# 👁️ AI Vision
Using small machine learning or AI models we will be performing various `Computer Vision` related operations, for example `Object Detection` on images. This project purely focused on cross-platform applications & we can run small AI/ML models on our mobile phones in offline mode.

> This project is buid on `kivy`, `kivymd` and uses `onnxruntime`, `numpy`, `opencv` etc. to perform the tasks. This is still at very early phase before this project matures at some level.

### Features
1. Objects detection from Image

## 📽️ Demo
To be added...

## 🖧 Our Scematic Architecture
To be added...

## 🧑‍💻 Quickstart Guide

### 📱 Download & Run the Android App
You can check the [Releases](https://github.com/daslearning-org/vision-ai/tags) and downlaod the latest version of the android app on your phone.

### 💻 Download & Run the Windows or Linux App
To be built later.

### 🐍 Run with Python

1. Clone the repo
```bash
git clone https://github.com/daslearning-org/vision-ai.git
```

2. Run the application
```bash
cd vision-ai/onnx/
pip install -r requirements.txt # virtual environment is recommended
python main.py
```

## 🦾 Build your own App
The Kivy project has a great tool named [Buildozer](https://buildozer.readthedocs.io/en/latest/) which can make mobile apps for `Android` & `iOS`

### 📱 Build Android App
A Linux environment is recommended for the app development. If you are on Windows, you may use `WSL` or any `Virtual Machine`. As of now the `buildozer` tool works on Python version `3.11` at maximum. I am going to use Python `3.11`

```bash
# add the python repository
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

# install all dependencies.
sudo apt install -y ant autoconf automake ccache cmake g++ gcc libbz2-dev libffi-dev libltdl-dev libtool libssl-dev lbzip2 make ninja-build openjdk-17-jdk patch patchelf pkg-config protobuf-compiler python3.11 python3.11-venv python3.11-dev

# optionally we can default to python 3.11
sudo ln -sf /usr/bin/python3.11 /usr/bin/python3
sudo ln -sf /usr/bin/python3.11 /usr/bin/python
sudo ln -sf /usr/bin/python3.11-config /usr/bin/python3-config

# optionally you may check the java installation with below commands
java -version
javac -version

# install python modules
git clone https://github.com/daslearning-org/vision-ai.git
cd vision-ai/onnx/
python3.11 -m venv .env # create python virtual environment
source .env/bin/activate
pip install -r req_android.txt

# build the android apk
buildozer android debug # this may take a good amount of time for the first time & will generate the apk in the bin directory
```
