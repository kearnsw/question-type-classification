#!/bin/bash

bash generate_configs.sh $1 $2

for ((n=0;n<$1;n++))
do
    bash train.sh configs/k-fold/$2_$n.json $3 
done
