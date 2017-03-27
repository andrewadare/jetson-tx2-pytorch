# Installing PyTorch on the NVIDIA Jetson TX1/TX2

[PyTorch](http://pytorch.org/) is a new deep learning framework that runs very well on the Jetson TX1 and TX2 boards. It is relatively simple and quick to install. Unlike TensorFlow, it requires no external swap partition to build on the TX1.

Although the TX2 has an ample 32 GB of eMMC, the TX1 has only half that, and it is easy to run out of space due to cruft from JetPack, Ubuntu packages, and installation artifacts. The cleanup section below lists ways to slim things down, and the steps here lean in the direction of minimalism.

The PyTorch developers recommend the Anaconda distribution. I was unable to find a recent Anaconda setup for ARM64, so I used the global python libraries.

## Scipy and LA libs
 - sudo apt install libopenblas-dev libatlas-dev liblapack-dev
 - sudo apt install liblapacke-dev checkinstall # For OpenCV
 - sudo -H pip3 install numpy scipy  # ~20-30 min

## Build tool prerequisites
 - pip3 install pyyaml
 - pip3 install scikit-build
 - sudo apt install ninja-build

## CMake
Check `cmake --version`. It looks like cmake >= 3.6 is required for the python bindings used in PyTorch. I followed http://askubuntu.com/a/865294 (and used --no-check-certificate) to get 3.7.2. The cmake executable was installed to /usr/local/bin. Make sure `cmake --version` reports the new version after installation.

## CFFI
 - sudo apt install python3-dev
 - sudo apt install libffi-dev
 - pip3 install cffi

## OpenCV
I followed a subset of these excellent [instructions](http://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/) for Python 3 from the pyimagesearch blog. I skipped the opencv_contrib modules to save space and build time. I also skipped CUDA and OpenCL integration, and went system-wide with the Python 3 bindings (no virtualenvs). Under the OpenCV source directory, I created build/ and ran this:
```
cmake \
    -D ENABLE_PRECOMPILED_HEADERS=OFF \
    -D WITH_OPENCL=OFF \
    -D WITH_CUDA=OFF \
    -D WITH_CUFFT=OFF \
    -D WITH_CUBLAS=OFF \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D INSTALL_C_EXAMPLES=OFF \
    -D PYTHON_EXECUTABLE=/usr/bin/python3.5 \
    -D BUILD_EXAMPLES=OFF ..
```
The first option significantly reduces the storage requirements during the build, at the expense of a slightly longer build time. Before discovering this option, I encountered errors like "failed to copy PCH file" on the TX1.

With this setup, OpenCV builds quickly--19 minutes on the TX2.

## CUDA and cuDNN
The Jetson does not ship ready to run deep learning models on the GPU. Unfortunately CUDA and cuDNN cannot simply be downloaded directly as for x86 systems; NVIDIA directs us to the JetPack SDK, which currently runs only on Ubuntu 14.04. In a VirtualBox instance, I [downloaded](https://developer.nvidia.com/embedded/downloads) the JetPack .run file. The default full installation is massive and takes hours to download, build, and flash. Moreover, it copies many GB of files to the local VM which is not what I want! I realized I could avoid the overkill and just select the CUDA and cuDNN packages, then cancel the installation once the .deb files are downloaded. Only the following 4 files are needed to install CUDA and cuDNN:
   1. cuda-repo-l4t-8-0-local_8.0.64-1_arm64.deb
   2. libcudnn5_5.1.10-1+cuda8.0_arm64.deb
   3. libcudnn5-dev_5.1.10-1+cuda8.0_arm64.deb
   4. cuda-l4t.sh

I created `~/cuda-l4t` on the Jetson and copied these 4 files there.
 - `sudo ./cuda-l4t.sh cuda-repo...arm64.deb 8.0 8-0` to install the CUDA libs.
 - To get cuDNN, do
`sudo dpkg -i <debfile>` for 2 and 3 above. Then do
```
sudo apt install -f
```
 - Check: `nvcc -V`
 - `ldconfig -p | grep cu` and `grep dnn` can be used to show the library locations. I also see /usr/include/cudnn.h

### cuDNN for PyTorch
Since tools/setup_helpers/cuda.py assumes /usr/local/cuda, CUDA_HOME need not be set. But the default cuDNN paths are not set for the TX1/TX2.

In pytorch/tools/setup_helpers/cudnn.py, there is a check for `os.getenv(CUDNN_LIB_DIR)` and another for the include dir. So I added the following to ~/.profile:*
```
export CUDNN_LIB_DIR=/usr/lib/aarch64-linux-gnu
export CUDNN_INCLUDE_DIR=/usr/include
```

## PyTorch source
```
git clone https://github.com/pytorch/pytorch.git
git checkout -b v0.1.10 v0.1.10
```

Now build PyTorch:
```
time python3 setup.py install # ~45 min on TX2. May need sudo (-E)
```

Test out in the python3 repl:
```
import torch
torch.backends.cudnn.is_acceptable(torch.cuda.FloatTensor(1))
```
If `True`, congratulations!

## Cleanup
The [postFlashTX1](https://github.com/jetsonhacks/postFlashTX1.git) repo contains some useful cleanup scripts. In addition:
```
sudo apt clean
sudo apt autoremove --purge
sudo rm /usr/src/*.tbz2 ## I had 6.9 GB of zip files
sudo rm /var/cuda-repo-8.0-local/*.deb
rm ~/temp # From my CMake 3.7 install
```
The OpenCV sources can also be removed if necessary.


## Footnotes
*If you see this:
```
[~]$ sudo python3 -c 'import os; print(os.getenv("CUDNN_LIB_DIR"))'
None
```
then cuDNN will fail to be included. Were the variables exported from ~/.bashrc instead of ~/.profile? Echoing at the prompt is not enough. Try this:
```
[~]$ sudo -E python3 -c 'import os; print(os.getenv("CUDNN_LIB_DIR"))'
/usr/lib/aarch64-linux-gnu
```
Exporting from ~/.profile is the preferred approach, but it requires a logout/login.
