echo Installs Prerequisites for OpenCV
sudo pip3 install virtualenv
cd tensorflow
source bin/activate

pip3 install --upgrade pip
sudo pip3 install --upgrade setuptools
sudo apt-get install libhdf5-dev libhdf5-serial-dev libhdf5-103
sudo apt-get install libqtgui4 libqtwebkit4 libqt4-test python3-pyqt5
sudo apt-get install libatlas-base-dev
sudo apt-get install libjasper-dev
pip install opencv-python

echo Installs Prerequisites for Tensorflow
sudo apt-get install gfortran
sudo apt-get install libhdf5-dev libc-ares-dev libeigen3-dev
sudo apt-get install libatlas-base-dev libopenblas-dev libblas-dev
sudo apt-get install liblapack-dev cython
sudo pip3 install pybind11
sudo pip3 install h5py
pip install tensorflow

pip install matplotlib
sudo apt-get install protobuf-compiler
pip3 install pydub
echo Prerequisites Downloaded Successfully

#export PYTHONPATH=$PYTHONPATH:/home/pi/tensorflow/models/research:/home/pi/tensorflow/models/research/slim" >> ~/.bashrc
#echo "source tensorflow/bin/activate" >> ~/.bashrc
