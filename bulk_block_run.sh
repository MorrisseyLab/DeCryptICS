#!/bin/bash

# get block list as variable
shopt -s nullglob
blocklist=(/home/doran/Work/images/Blocks/block*/input_files.txt)

for i in "${blocklist[@]}"
do
	python ./run_script.py count "$i" -r
done
