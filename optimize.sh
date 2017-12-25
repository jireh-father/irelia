#!/bin/sh
nohup python -u optimizer.py --dataset_dir=/home/irelia/ --save_dir=/home/irelia/checkpoint --epoch=25 --batch_size=128 --optimizer=rmsprop --shuffle_buffer_size=10000 > opt.log &