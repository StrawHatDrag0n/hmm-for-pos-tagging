import argparse
from tri_hmm import TriHMM
from file_utils import read_test_data, write_predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data file path')
    parser.add_argument('data_file_path', type=str,
                        help='data file path')
    args = parser.parse_args()
    data = read_test_data(args.data_file_path)
    clf = TriHMM.load_model('hmmmodel.txt')
    predictions = clf.predict(data)
    write_predictions(predictions)
