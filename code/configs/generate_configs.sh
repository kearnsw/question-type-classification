#!/bin/bash

for ((n=0;n<$1;n++))
do
    sed "s|placeholder|$n|" $2.json > k-fold/$2_$n.json
done
