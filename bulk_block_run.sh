#!/bin/bash

# get block list as variable
shopt -s nullglob
#blocklist=(/full/path/to/block/folders/block*/input_files.txt), eg:
blocklist=(/home/doran/Work/images/Leeds_May2019/KM*/input_files.txt)

for i in "${blocklist[@]}"
do
	python ./run_script.py count "$i" -q block_analysis
done
