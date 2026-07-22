#!/bin/bash
set -e

echo "======================================"
echo " Setting up HaMeR on fresh GPU VM..."
echo "======================================"

echo "[1/4] Installing system dependencies..."
sudo apt-get update -y
sudo apt-get install -y python3-venv python3-pip git wget unzip libgl1 libglib2.0-0

echo "[2/4] Setting up Python virtual environment..."
python3 -m venv .hamer
source .hamer/bin/activate
pip install --upgrade pip
pip install gdown

echo "[3/4] Installing PyTorch and HaMeR dependencies..."
# Using cu118 or cu121 depending on standard setups, cu117 is from standard docs but cu118/121 is usually safer on L4.
# Falling back to the repo's recommended cu117
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation
pip install -e .[all]
pip install -v -e third-party/ViTPose

echo "[4/4] Downloading models and data..."
# 1. Download demo data
bash fetch_demo_data.sh

# 2. Download MANO weights from user's Drive
mkdir -p _DATA/data/mano
echo "Downloading MANO weights (ZIP)..."
gdown 1VieuI7JEWvZiMjxTyu8WJmKoFrZhVIGK -O mano_temp.zip
unzip -o mano_temp.zip -d mano_extracted
# Find MANO_RIGHT.pkl inside the extracted folder and move it
find mano_extracted -name "MANO_RIGHT.pkl" -exec mv {} _DATA/data/mano/ \;
rm -rf mano_temp.zip mano_extracted

# 3. Download the test video
echo "Downloading test video..."
gdown 1BHWhSPspZsSGQvAS3dEg4UiLawDYL7ZO -O test_video.mp4

echo "======================================"
echo " Setup Complete! 🎉"
echo " "
echo " You can now run the video test by executing:"
echo "   source .hamer/bin/activate"
echo "   python process_video.py --video_path test_video.mp4 --out_video output_30s.mp4 --max_seconds 30"
echo "======================================"
