import pandas as pd
import os
from argparse import ArgumentParser
from typing import List, Tuple
from math import floor

data_dir = os.path.dirname(__file__)

def parse_args():
    cli_parser = ArgumentParser()
    cli_parser.add_argument("-d", "--data-dir", default="raw", help="file containing data files")
    cli_parser.add_argument("-n", "--folds", default=10, help="number of folds")
    cli_parser.add_argument("-g", "--gard-only", action='store_true')
    cli_parser.add_argument("-y", "--yahoo_only", action='store_true')
    return cli_parser.parse_args()


def create_folds(data: pd.DataFrame, nb_folds: int, test_set: bool=True, equal_folds: bool=True) -> Tuple:
    # remove data from fields not in original gard dataset
    new_fields = ["na", "summary", "websearch", "WebSearch"]
    data = data[~data["QT"].isin(new_fields)]

    # Remove unannotated examples and randomize with a seed for reproducibility
    data = data[(data["QT"].isnull() == False)].sample(frac=1, random_state=1986).reset_index(drop=True)
    nb_examples = len(data)
    # Take 10% as independent test set
    if test_set:
        test = data.loc[nb_examples*.9:]
        train = data.loc[:nb_examples*.9]
    else:
        test = None
        train = data

    fold_size = floor(len(train)/nb_folds)
    folds = [[]] * nb_folds

    if equal_folds:
        for k in range(nb_folds):
            folds[k] = train[k*fold_size:(k+1)*fold_size]
    else:
        for k in range(nb_folds-1):
            folds[k] = train[k*fold_size:(k+1)*fold_size]
        # Add the remaining samples to the last batch
        folds[nb_folds-1] = data[-fold_size:]

    return folds, test


def get_files(dir: str) -> List:
    dir = os.path.join(data_dir, dir)

    for path, _, fns in os.walk(dir):
        for fn in fns:
            yield os.sep.join([path, fn])


def write_fold(folds: List, idx: int) -> None:
    dev = folds[idx]
    dev.to_csv(os.path.join(data_dir, "dev/dev_{0}".format(idx)), sep="\t")

    train: pd.DataFrame = pd.concat([df for i, df in enumerate(folds) if i != idx])
    train.to_csv(os.path.join(data_dir, "train/train_{0}".format(idx)), sep="\t")


def write_folds(folds: List) -> None:
    nb_folds = len(folds)
    for i in range(nb_folds):
        write_fold(folds, i)


if __name__ == "__main__":
    args = parse_args()
    if args.gard_only:
        print("Generating {0} datafiles from GARD data only.".format(args.folds))
        annotations = pd.read_csv(os.path.join(data_dir, "raw/GARD.tsv"), delimiter="\t")
    elif args.yahoo_only:
        print("Generating {0} datafiles from Yahoo data only.".format(args.folds))
        annotations = pd.read_csv(os.path.join(data_dir, "raw/yahoo_answers_qt.tsv"), delimiter="\t")
    else:
        print("Generating {0} datafile from all data".format(args.folds))
        annotations = [pd.read_csv(fn, delimiter="\t") for fn in get_files(args.data_dir)]
        annotations: pd.DataFrame = pd.concat(annotations)
    folds, test = create_folds(annotations, int(args.folds))
    write_folds(folds)
    if test is not None:
        test.to_csv(os.path.join(data_dir, "test/test.tsv"), sep="\t")
