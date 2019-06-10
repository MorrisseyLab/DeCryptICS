#!/bin/bash

# get block list as variable
shopt -s nullglob
#blocklist=(/full/path/to/block/folders/blocknames*/), eg:
blocklist=(/home/doran/Work/images/Leeds_May2019/KM*/)

for i in "${blocklist[@]}"
do
	python ./generate_block_list.py "$i"
done
