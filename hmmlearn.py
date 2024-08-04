import argparse

from file_utils import read_training_data
from tri_hmm import TriHMM

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data file path')
    parser.add_argument('data_file_path', type=str,
                        help='data file path')
    args = parser.parse_args()
    data = read_training_data(args.data_file_path)
    clf = TriHMM(use_log=True)
    clf.fit(data)
    clf.save_model('hmmmodel.txt')
