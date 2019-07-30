#!/usr/bin/env bash

python train.py -level info -ep 10
python train.py -level info -ep 10 -bs 32

cmd /k