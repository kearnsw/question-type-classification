#!/bin/bash

ID=$(echo $1 | grep -o "[0-9]")
DIRNAME=$(grep -oP "\w*(?=.txt.gz)" $1 | awk '{print}' ORS='-' | rev | cut -c 2- | rev)
python -m allennlp.run train $1 -s models/$2/$DIRNAME$ID --include-package sc
