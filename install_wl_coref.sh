#!/bin/bash
conda create -y --name wl-coref python=3.7 openjdk perl && conda activate wl-coref
python -m pip install -r requirements.txt