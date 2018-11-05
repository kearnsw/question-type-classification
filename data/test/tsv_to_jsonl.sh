#!/bin/bash

tail -n +2 $1 | sed "s/\"/'/g" | awk -F "\t" '{print "{\"QT\": \"" $2 "\", \"Question\": \""$3 "\"}"}'

