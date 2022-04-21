#!/bin/bash
pip install gdown -q
gdown https://drive.gooigle.com/uc?id=1M_FQGAWtRP_XYx0K43kTn73kcOwZw1Jj
mkdir checkpoint
unzip model_rk.zip
rm model_rk.zip