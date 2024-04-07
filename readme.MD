Tạo môi trường ảo:
sudo pip3 install virtualenv
cd tensorflow
source bin/activate
echo "source tensorflow/bin/activate" >> ~/.bashrc

Cài đặt các thư viện:
run bash setup.sh

Cài đặt model:
run bash add_model.sh

Liên kết model:
export PYTHONPATH=$PYTHONPATH:/home/pi/tensorflow/models/research:/home/pi/tensorflow/models/research/slim" >> ~/.bashrc