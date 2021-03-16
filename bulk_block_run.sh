#!/bin/bash

# get block list as variable
shopt -s nullglob

#blocklist=(/full/path/to/block/folders/block*/input_files.txt)
blocklist=(/home/doran/Work/images/Leeds_May2019/splitbyKM/newbatch_18Feb2021/KM*/input_files.txt)
#blocklist=(/home/doran/Work/images/Serial_blocks_Oct2019/block*/input_files.txt)
#blocklist=(/home/doran/Work/images/KRAS_study/block*/input_files.txt)

for i in "${blocklist[@]}"
do
	python ./run_script.py count "$i" -q block_analysis
done
