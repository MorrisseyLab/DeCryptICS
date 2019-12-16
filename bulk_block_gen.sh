#!/bin/bash

# get block list as variable
shopt -s nullglob
#blocklist=(/full/path/to/block/folders/blocknames*/)
blocklist=(/home/doran/Work/images/Leeds_May2019/splitbyKM/newbatch_25Nov/KM*/)
#blocklist=(/home/doran/Work/images/Serial_blocks_Oct2019/block*/)

for i in "${blocklist[@]}"
do
	python ./generate_block_list.py "$i"
done
