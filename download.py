from pyroomacoustics.datasets.cmu_arctic import CMUArcticCorpus
from pathlib import Path 
from argparse import ArgumentParser

def main(output_dir):
    CMUArcticCorpus(output_dir,download=True)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("output_dir", type=Path, help="Downloading data output path.")
    args = parser.parse_args()
    main(**vars(args))