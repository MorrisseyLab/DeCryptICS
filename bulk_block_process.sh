#!/bin/bash

# get block list as variable
shopt -s nullglob

# write uncommented variable assignment of form:
#blocklist=(/full/path/to/block/folders/block*/input_files.txt)

for i in "${blocklist[@]}"
do
	python ./process_output.py "$i" -c 0.5 -p 0.5 -f 0.25 -r
done
