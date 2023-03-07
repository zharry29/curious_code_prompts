import json
import argparse
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path', type=str, required=True
    )
    return parser.parse_args()


def main():
    args = get_args()
    data = pd.read_csv(args.data_path)
    task_name = data.iloc[:, 0]
    performances = data.iloc[:, 1:]

    assert len(task_name) == len(performances)

    for i in range(len(task_name)):
        print(task_name[i])
        print(performances.iloc[i, :].rank())
    


if __name__ == '__main__':
    main()


