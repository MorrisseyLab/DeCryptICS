#!/bin/bash

# get block list as variable
shopt -s nullglob
# write uncommented variable assignment of form:
#blocklist=(/full/path/to/block/folders/block*/input_files.txt)

for i in "${blocklist[@]}"
do
	python ./run_script.py count "$i" -q block_analysis
done
