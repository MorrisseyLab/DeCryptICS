#!/bin/bash

# get block list as variable
shopt -s nullglob

#blocklist=(/home/doran/Work/images/Leeds_May2019/curated_cryptdata/train/KM*/input_files.txt)
blocklist=(/home/doran/Work/images/Anne-Claire_curated_2021/HR*STAG2/input_files.txt)

for i in "${blocklist[@]}"
do
	python ./run_script.py read "$i" -q block_analysis
done
