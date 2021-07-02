#!/bin/bash

# get block list as variable
shopt -s nullglob
# write uncommented variable assignment of form:
#blocklist=(/full/path/to/block/folders/blocknames*/)

for i in "${blocklist[@]}"
do
	python ./generate_block_list.py "$i"
done
