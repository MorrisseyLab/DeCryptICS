#!/bin/bash

# get block list as variable
shopt -s nullglob
# write uncommented variable assignment of form:
#blocklist=(/path/to/blocks*/input_files.txt)

for i in "${blocklist[@]}"
do
	python ./run_script.py read "$i" -q block_analysis
done
