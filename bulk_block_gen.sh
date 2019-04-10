#!/bin/bash

# get block list as variable
shopt -s nullglob
blocklist=(/home/doran/Work/images/Blocks/block*/)

for i in "${blocklist[@]}"
do
	python ./generate_block_list.py "$i"
done
