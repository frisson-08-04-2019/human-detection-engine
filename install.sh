#!/bin/bash

conda create -n human-detection-engine pip python=3.6 -y
source activate human-detection-engine
pip install -r requirements.txt
