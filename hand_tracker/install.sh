#!/bin/bash
set -e

echo "=================================================="
echo " Setting up 3D Hand Tracker Integration Package..."
echo "=================================================="

# 1. Install system headless rendering dependencies
echo "[1/4] Installing system headless rendering dependencies..."
sudo apt-get update -y
sudo apt-get install -y python3-venv python3-pip git wget unzip ffmpeg libgl1 libglib2.0-0 libegl1 libegl-mesa0 libgl1-mesa-dri libxrender1 libxext6 libsm6 libosmesa6 freeglut3-dev xvfb

# Add 16GB swap file if needed to prevent build crashes
if [ ! -f /swapfile ]; then
    echo "Creating 16GB swap file..."
    sudo fallocate -l 16G /swapfile || true
    sudo chmod 600 /swapfile || true
    sudo mkswap /swapfile || true
    sudo swapon /swapfile || true
fi

# 2. Setup Virtual Environment
echo "[2/4] Setting up Python dependencies..."
python3 -m venv .venv_hand_tracker
source .venv_hand_tracker/bin/activate
pip install --upgrade pip
pip install "setuptools<70" "numpy<2.0.0" "pyglet<2.0.0" wheel ninja gdown

# Install PyTorch CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install Detectron2, Chumpy, MMCV, ViTPose
MAX_JOBS=1 pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation
MAX_JOBS=1 pip install 'git+https://github.com/mattloper/chumpy' --no-build-isolation
MAX_JOBS=1 pip install mmcv --no-build-isolation
MAX_JOBS=1 pip install -e .[all] --no-build-isolation
MAX_JOBS=1 pip install -v -e third-party/ViTPose

# 3. Download Model Checkpoints & MANO Weights
echo "[3/4] Downloading HaMeR & MANO model weights..."
bash fetch_demo_data.sh
rm -f hamer_demo_data.tar.gz

mkdir -p _DATA/data/mano
echo "Downloading MANO weights..."
gdown 1VieuI7JEWvZiMjxTyu8WJmKoFrZhVIGK -O mano_temp.zip
unzip -o mano_temp.zip -d mano_extracted
find mano_extracted -name "MANO_RIGHT.pkl" -exec mv {} _DATA/data/mano/ \;
rm -rf mano_temp.zip mano_extracted

echo "=================================================="
echo " Hand Tracker Installation Complete! 🎉"
echo " "
echo " Run the test script:"
echo "   source .venv_hand_tracker/bin/activate"
echo "   python hand_tracker/example_usage.py"
echo "=================================================="
