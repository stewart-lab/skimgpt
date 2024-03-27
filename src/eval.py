import pandas as pd
import numpy as np
import argparse
from glob import glob
from functools import partial

def main():
	parser = argparse.ArgumentParser("arg_parser")
	parser.add_argument("-output_dir", type=str, required=True)
	args = parser.parse_args()
 
	files = glob(f"{args.output_dir}/filtered/cot_*")
	files.sort()
	total_tsv = pd.concat(map(partial(pd.read_csv, sep = "\t"), files), ignore_index=True)
 
	total_tsv.to_csv(f"{args.output_dir}/full_output.tsv", sep='\t')
	
	ab = [eval(l) for l in list(total_tsv["ab_score"])]
	bc = [eval(l) for l in list(total_tsv["bc_score"])]

	total = [ab[i] + bc[i] for i in range(len(ab))]
	y_hat = []
	for pred in total:
		y_hat.extend(pred)

	with open("scores.txt") as f:
		scores = f.read().split("\n")[:-1]
	y = [eval(i) for i in scores]
 
	y_hat = np.array(y_hat)
 
	print(y_hat)
	y = np.array(y)
	
	acc = (y_hat == y).sum() / len(y_hat)
 
	print(f"Relevance accuracy is: {acc}")

if __name__ == "__main__":
	main()