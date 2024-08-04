import argparse
from collections import defaultdict, OrderedDict


def read_training_data(data_path: str):
    return read_training_data_v1(data_path)


def read_training_data_v1(data_path: str):
    data = defaultdict(lambda: {'words': list(), 'tags': list()})
    with open(data_path, encoding="utf8") as data_file:
        lines = data_file.readlines()
        for idx, line in enumerate(lines):
            words_tags = line.split()
            for word_tag in words_tags:
                word, tag = word_tag.rsplit('/', 1)
                data[idx]['words'].append(word)
                data[idx]['tags'].append(tag)
    return data


def read_test_data(data_path):
    return read_test_data_v1(data_path)


def read_test_data_v1(data_path):
    data = OrderedDict()
    with open(data_path, encoding='utf8') as data_file:
        lines = data_file.readlines()
        for idx, line in enumerate(lines):
            words = line.split()
            for word in words:
                if data.get(idx):
                    data[idx].append(word)
                else:
                    data[idx] = [word]
    return data


def write_predictions(predictions, file_path='hmmoutput.txt'):
    with open(file_path, 'w+', encoding='utf8') as file:
        data = []
        for idx, prediction in predictions.items():
            words = prediction.get('words')
            tags = prediction.get('tags')
            sentence = ''
            for word, tag in zip(words, tags):
                sentence += f'{word}/{tag or ""} '
            sentence += '\n'
            data.append(sentence)
        file.writelines(data)


def display_data(data):
    for i, datum in data.items():
        print(i)
        for k, v in datum.items():
            print(k, v)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data file path')
    parser.add_argument('data_file_path', type=str,
                        help='data file path')
    args = parser.parse_args()
    data = read_training_data(args.data_file_path)

