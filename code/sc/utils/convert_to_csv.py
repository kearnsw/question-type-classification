import json
import sys

json_data = []
with open(sys.argv[1]) as f:
	for line in f:
		json_data.append(json.loads(line))

with open(sys.argv[2]) as f:
	f.write("True\tPred\tQuestion\n")
	for d in json_data:
		f.write("{0}\t{1}\t{2}\n".format(d["true"], d["pred"], d["question"]))

