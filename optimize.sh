#!/bin/sh
nohup python -u optimizer.py --epoch=50 --dataset_dir=/home/data/irelia --model_file_name=dataset_real.csv > opt.log &