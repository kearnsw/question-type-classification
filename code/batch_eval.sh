#!/bin/bash

dirs=$(ls -d $1/*)

echo $1

for dir in $dirs;
do
	./eval.sh $dir/model.tar.gz | grep accuracy
done
