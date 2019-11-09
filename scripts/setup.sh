# cd ~
# wget https://github.com/lamhoangtung/kaggle-kernel-setup/raw/master/install_common.sh
# chmod 777 install_common.sh
# ./install_common.sh
# source .bashrc
# tmux

echo "Clone the code"
git clone https://github.com/lamhoangtung/Semantic-Segmentation-Suite
cd Semantic-Segmentation-Suite
git checkout tiny_cityscape
echo "Install dependencies"
# python3 -m pip install -r requirements.txt
echo "Download and preprecess the training data"
gdrive download 1UB81Wx44nYOBLzQckwrKAu3wkgnisqW9
unzip -qq apolo_city_scape.zip
chmod 777 ./scripts/*.sh
echo "Done. Run ./scripts/train_mobile-net-v2_mobile-unet-skip.sh to start training!"