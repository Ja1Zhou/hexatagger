#!/usr/bin/env bash
conda env create -n hexa python=3.10 -y
conda activate hexa
conda install -c conda-forge rust -y
pip install -r requirements.txt