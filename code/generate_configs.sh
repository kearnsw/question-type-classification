#!/bin/bash

echo "Generating $1 configs into configs/k-fold ..."
for ((n=0;n<$1;n++))
do
    sed "s|placeholder|$n|" configs/$2.json > configs/k-fold/$2_$n.json
done
