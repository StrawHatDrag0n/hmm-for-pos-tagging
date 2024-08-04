import argparse
import json
import numpy as np
import math
from collections import defaultdict, OrderedDict

from file_utils import read_training_data

START_TAG = '<S>'
LARGE_NEGATIVE = -1e8


class HMM(object):
    def __init__(self, alpha=0.1, use_log=False):
        self.alpha = alpha
        self.use_log = use_log

        self.emission = defaultdict(lambda: defaultdict(float))
        self.transition = defaultdict(lambda: defaultdict(float, {START_TAG: 0.0}))
        self.word_to_tag = defaultdict(lambda: defaultdict(float))
        self.tag_count = defaultdict(float)
        self.tag_to_vocab = defaultdict(set)
        self.tags = set()
        self.most_common_tags = list()

    def _set_params(self, params):
        self.alpha = params.get('alpha')
        self.use_log = params.get('use_log')
        self.emission = params.get('emission')
        self.transition = params.get('transition')
        self.tags = params.get('tags')
        self.tag_count = params.get('tag_count')
        self.most_common_tags = params.get('most_common_tags')

    def calculate_corpus_statistics(self, data):
        for _, words_tags_data in data.items():
            words = words_tags_data.get('words')
            tags = words_tags_data.get('tags')
            for i, word_tag in enumerate(zip(words, tags)):
                word, tag = word_tag
                self.tag_to_vocab[tag].add(word)
                if i == 0:
                    self.transition[tag][START_TAG] += 1
                    self.tag_count[START_TAG] += 1
                else:
                    self.transition[tag][tags[i-1]] += 1
                self.word_to_tag[word][tag] += 1
                self.tag_count[tag] += 1
                self.tags.add(tag)

    def calculate_emission(self, data):
        for word, tags in self.word_to_tag.items():
            for tag in self.tags:
                self.emission[word][tag] = self.word_to_tag[word].get(tag, 0) / self.tag_count[tag]

    def calculate_transition(self):
        for tag in self.tags:
            self.transition[tag][START_TAG] += self.alpha
            self.transition[tag][START_TAG] /= (self.tag_count[START_TAG] + self.alpha * len(self.tags))
        for tag in self.tags:
            for prev_tag in self.tags:
                self.transition[tag][prev_tag] += self.alpha
                self.transition[tag][prev_tag] /= (self.tag_count[prev_tag] + self.alpha * len(self.tags))

    def calculate_most_common_tags(self):
        tag_counts = [(len(vocab), tag) for tag, vocab in self.tag_to_vocab.items() if tag != START_TAG]
        tag_counts.sort(reverse=True)
        self.most_common_tags = [tag for count, tag in tag_counts[:]]

    def fit(self, data):
        self.calculate_corpus_statistics(data)
        self.calculate_transition()
        self.calculate_emission(data)
        self.calculate_most_common_tags()

    def save_model(self, model_path):
        model_parameters = dict()
        model_parameters['alpha'] = self.alpha
        model_parameters['use_log'] = self.use_log
        model_parameters['transition'] = self.transition
        model_parameters['emission'] = self.emission
        model_parameters['tags'] = list(self.tags)
        model_parameters['tag_count'] = self.tag_count
        model_parameters['most_common_tags'] = self.most_common_tags
        with open(model_path, 'w+') as model_file:
            model_file.write(str(json.dumps(model_parameters)))

    def predict(self, data):
        result = OrderedDict()
        tag_to_idx = {tag: idx for idx, tag in enumerate(self.tags)}
        idx_to_tags = {idx: tag for idx, tag in enumerate(self.tags)}

        for si, words in data.items():
            viterbi = np.zeros((len(self.tags), len(words)))
            back_pointer = np.zeros((len(self.tags), len(words)))
            word = words[0]
            for tag in self.tags:
                tag_idx = tag_to_idx.get(tag)
                transition_prob = self.transition.get(tag, dict()).get(START_TAG)
                emission_prob = self.emission.get(word, dict()).get(tag)

                if self.use_log:
                    log_transition_prob = math.log(transition_prob)
                    log_emission_prob = math.log(emission_prob) \
                        if emission_prob is not None and emission_prob != 0.0 else LARGE_NEGATIVE
                    prob = log_transition_prob + log_emission_prob
                else:
                    transition_prob = self.transition.get(tag, dict()).get(START_TAG)
                    emission_prob = emission_prob if emission_prob is not None else 1.0
                    prob = transition_prob * emission_prob

                viterbi[tag_idx][0] = prob
                back_pointer[tag_idx][0] = 0

            for idx, word in enumerate(words[1:], start=1):
                for tag_idx, tag in enumerate(self.tags):
                    if self.emission.get(word, dict()).get(tag) is None:
                        if self.use_log:
                            log_probs = [
                                viterbi[tag_to_idx[prev_tag]][idx - 1] + math.log(self.transition[tag][prev_tag])
                                for prev_tag in self.most_common_tags]
                            log_transition_prob = max(log_probs)
                            most_common_tag_idx = log_probs.index(log_transition_prob)
                            prob = log_transition_prob
                            next_id = tag_to_idx.get(self.most_common_tags[most_common_tag_idx])
                        else:
                            probs = [
                                viterbi[tag_to_idx[prev_tag]][idx - 1] * self.transition[tag][prev_tag]
                                for prev_tag in self.most_common_tags]
                            transition_prob = max(probs)
                            most_common_tag_idx = probs.index(transition_prob)
                            prob = transition_prob
                            next_id = tag_to_idx.get(self.most_common_tags[most_common_tag_idx])

                        viterbi[tag_idx][idx] = prob
                        back_pointer[tag_idx][idx] = next_id
                    else:
                        if self.use_log:
                            emission_prob = self.emission.get(word, dict()).get(tag)
                            log_emission_prob = math.log(emission_prob) \
                                if emission_prob is not None and emission_prob != 0.0 else LARGE_NEGATIVE
                            log_probs = [
                                viterbi[tag_to_idx[prev_tag]][idx - 1] + math.log(self.transition[tag][prev_tag])
                                for prev_tag in self.tags]
                            log_transition_prob = max(log_probs)
                            prob = log_transition_prob + log_emission_prob
                            next_id = log_probs.index(log_transition_prob)
                        else:
                            emission_prob = self.emission.get(word, dict()).get(tag, 1)
                            probs = [viterbi[tag_to_idx[prev_tag]][idx-1] * self.transition[tag][prev_tag] for prev_tag in self.tags]
                            transition_prob = max(probs)
                            prob = transition_prob * emission_prob
                            next_id = probs.index(transition_prob)

                        viterbi[tag_idx][idx] = prob
                        back_pointer[tag_idx][idx] = next_id

            best_tag_idx = int(np.argmax(viterbi[:, len(words)-1]))
            tag_sequence = []
            for idx in reversed(range(len(words))):
                tag_sequence.append(idx_to_tags[best_tag_idx])
                best_tag_idx = int(back_pointer[best_tag_idx][idx])
            result[si] = {'words': words, 'tags': tag_sequence[::-1]}
        return result

    @classmethod
    def load_model(cls, model_path):
        clf = cls()
        with open(model_path) as model:
            model_parameters = json.loads(model.read())
            clf._set_params(model_parameters)
        return clf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data file path')
    parser.add_argument('data_file_path', type=str,
                        help='data file path')
    args = parser.parse_args()
    data = read_training_data(args.data_file_path)
    clf = HMM()
    clf.fit(data)
