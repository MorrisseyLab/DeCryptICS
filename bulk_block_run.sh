#!/bin/bash

# get block list as variable
shopt -s nullglob
#blocklist=(/full/path/to/block/folders/block*/input_files.txt)
blocklist=(/home/doran/Work/images/mPAS_blocks/block*/input_files.txt)

for i in "${blocklist[@]}"
do
	python ./run_script.py count "$i" -q block_analysis
done
