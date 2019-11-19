@echo off
PATH = %PATH%;%USERPROFILE%\Miniconda3\Scripts
conda create -n human-detection-engine pip python=3.6 -y
call activate human-detection-engine
pip install -r requirements.txt
