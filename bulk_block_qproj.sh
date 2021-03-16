#!/bin/bash

# get block list as variable
shopt -s nullglob

blocklist=(/home/doran/Work/images/Leeds_May2019/curated_cryptdata/test/KM*/input_files.txt)

for i in "${blocklist[@]}"
do
	python ./run_script.py read "$i" -q block_analysis
done
