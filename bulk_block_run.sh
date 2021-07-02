#!/bin/bash

# get block list as variable
shopt -s nullglob

#blocklist=(/full/path/to/block/folders/block*/input_files.txt)
#blocklist=(/home/doran/Work/images/Anne-Claire_curated_2021/HR*/input_files.txt)
blocklist=(/home/doran/Work/images/Leeds_May2019/curated_cryptdata/test/KM*/input_files.txt)

for i in "${blocklist[@]}"
do
	python ./run_script.py count "$i" -q v2
done
