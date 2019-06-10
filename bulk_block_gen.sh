#!/bin/bash

# get block list as variable
shopt -s nullglob
#blocklist=(/full/path/to/block/folders/blocknames*/):

for i in "${blocklist[@]}"
do
	python ./generate_block_list.py "$i"
done
