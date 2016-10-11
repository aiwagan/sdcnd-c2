ln -s /sharefolder .
ln -s sharefolder/opencv .
ln /dev/null /dev/raw1394
cp /sharefolder/theanorc .theanorc
cp /sharefolder/dnn.py /usr/local/lib/python2.7/dist-packages/theano/sandbox/cuda/dnn.py
apt-get install python3 python3-dev python3-numpy
cd /sharefolder/opencv/
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") -D PYTHON_EXECUTABLE=$(which python3) .
make -j8
cd lib/python3/
ln -s cv2.cpython-34m.so cv2.so 
cd /sharefolder/opencv/
make install
cd
curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py
pip3 install jupyter
ipython kernel install
pip3 install matplotlib
pip3 install pillow
pip3 install moviepy 

python3 -c "import tensorflow; print(tensorflow.__version__);"
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.11.0rc0-cp34-cp34m-linux_x86_64.whl
pip3 install --upgrade $TF_BINARY_URL
python3 -c "import tensorflow; print(tensorflow.__version__);"
pip3 install --upgrade scipy
echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1404/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list
apt-get update && apt-get install -y --no-install-recommends libcudnn5-dev=5.1.3-1+cuda7.5

sh run_jupyter.sh
