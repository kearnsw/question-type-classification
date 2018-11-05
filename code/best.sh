#!/bin/bash
grep "validation_accuracy" $1/*/python_logging.log |  grep "[0-9]\.[0-9]*" 

