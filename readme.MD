Tạo môi trường ảo:

apt install python3.10-venv
sudo pip3 install virtualenv
 python3 -m venv tensorflow

cd tensorflow

source bin/activate

echo "source tensorflow/bin/activate" >> ~/.bashrc

Cài đặt các thư viện:

run bash setup.sh

Cài đặt model:

run bash add_model.sh

Liên kết model:

export PYTHONPATH=$PYTHONPATH:/home/pi/tensorflow/models/research:/home/pi/tensorflow/models/research/slim" >> ~/.bashrc

Tạo thư mục chứa model:
mkdir od-models
cd od-models

Tải model:
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz

Giải nén model:
tar -xvf ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz
mv ssd_mobilenet_v2_320x320_coco17_tpu-8 my_mobilenet_model

Về thư mục chính và thử nghiệm:
cd ~/tensorflow
python3 main.py

