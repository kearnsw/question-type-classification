#!/bin/bash

python -m allennlp.run evaluate $1 --evaluation-data-file "../data/test/test.tsv" --include-package sc 

