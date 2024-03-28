#!/bin/bash
# Comand to install all dependences
git clone https://github.com/IDEA-Research/GroundingDINO.git
pip install -e ./GroundingDINO
pip install -r requirements.txt
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth -P GroundingDINO_weights


