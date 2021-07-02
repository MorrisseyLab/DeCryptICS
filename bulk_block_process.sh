#!/bin/bash

# get block list as variable
shopt -s nullglob

#blocklist=(/full/path/to/block/folders/block*/input_files.txt)
#blocklist=(/home/doran/Work/images/Anne-Claire_curated_2021/HR*/input_files.txt)
blocklist=(/home/doran/Work/images/Leeds_May2019/curated_cryptdata/test/KM*/input_files.txt)

for i in "${blocklist[@]}"
do
	python ./process_output.py "$i" -c 0.5 -p 0.5 -f 0.25 -g 0.5 -r
done
