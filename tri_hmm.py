import json
import numpy as np
import math
from collections import defaultdict, OrderedDict

START_TAG = '<S>'
LARGE_NEGATIVE = -1e8


class TriHMM(object):
    def __init__(self, alpha=0.1, use_log=False):
        self.alpha = alpha
        self.use_log = use_log

        self.emission = defaultdict(lambda: defaultdict(float))
        self.transition = defaultdict(lambda: defaultdict(lambda: defaultdict(float, {START_TAG: 0.0})))
        self.word_to_tag = defaultdict(lambda: defaultdict(float))

        self.tag_count = defaultdict(float)
        self.tag_tag_count = defaultdict(lambda: defaultdict(float))

        self.tags = set()
        self.tag_to_idx = dict()
        self.idx_to_tag = dict()
        self.most_common_tags = list()

    def calculate_corpus_statistics(self, data):
        for _, words_tags_data in data.items():
            words = words_tags_data.get('words')
            tags = words_tags_data.get('tags')
            words_tags = list(zip(words, tags))
            for idx, word_tag in enumerate(words_tags[1:]):
                word, tag = word_tag
                if idx == 0:
                    self.transition[tag][START_TAG][START_TAG] += 1
                    self.tag_tag_count[START_TAG][START_TAG] += 1
                elif idx == 1:
                    self.transition[tag][tags[idx-1]][START_TAG] += 1
                    self.tag_tag_count[tags[idx-1]][START_TAG] += 1
                else:
                    self.transition[tag][tags[idx-1]][tags[idx-2]] += 1

                self.word_to_tag[word][tag] += 1
                self.tag_count[tag] += 1
                self.tag_tag_count[tag][tags[idx-1]] += 1
                self.tags.add(tag)
        self.tags = list(self.tags)
        self.tag_to_idx = {tag: idx for idx, tag in enumerate(self.tags)}
        self.idx_to_tag = {idx: tag for idx, tag in enumerate(self.tags)}

    def calculate_transition(self):
        for tag in self.tags:
            self.transition[tag][START_TAG][START_TAG] += self.alpha
            self.transition[tag][START_TAG][START_TAG] /= (
                        self.tag_tag_count[START_TAG][START_TAG] + self.alpha * len(self.tags))
            for prev_tag in self.tags:
                self.transition[tag][prev_tag][START_TAG] += self.alpha
                self.transition[tag][prev_tag][START_TAG] /= (self.tag_tag_count[prev_tag][START_TAG] + self.alpha * len(self.tags))

        for tag in self.tags:
            for prev_tag in self.tags:
                for prev_prev_tag in self.tags:
                    self.transition[tag][prev_tag][prev_prev_tag] += self.alpha
                    self.transition[tag][prev_tag][prev_prev_tag] /= (self.tag_tag_count[prev_tag][prev_prev_tag] + self.alpha * len(self.tags))

    def calculate_emission(self, data):
        for word, tags in self.word_to_tag.items():
            for tag in self.tags:
                self.emission[word][tag] = self.word_to_tag[word].get(tag, 0) / self.tag_count[tag]

    def calculate_most_common_tags(self):
        self.most_common_tags = list(self.tags)

    def fit(self, data):
        self.calculate_corpus_statistics(data)
        self.calculate_transition()
        self.calculate_emission(data)
        self.calculate_most_common_tags()

    def save_model(self, model_path):
        model_parameters = dict()
        model_parameters['tag_to_idx'] = self.tag_to_idx
        model_parameters['idx_to_tag'] = self.idx_to_tag
        model_parameters['transition'] = self.transition
        model_parameters['emission'] = self.emission
        model_parameters['tags'] = list(self.tags)
        model_parameters['tag_count'] = self.tag_count
        model_parameters['tag_tag_count'] = self.tag_tag_count
        model_parameters['most_common_tags'] = self.most_common_tags
        with open(model_path, 'w+') as model_file:
            model_file.write(str(json.dumps(model_parameters)))

    def _set_params(self, params):
        self.tag_to_idx = {k: int(v) for k, v in params.get('tag_to_idx', dict()).items()}
        self.idx_to_tag = {int(k): v for k, v in params.get('idx_to_tag', dict()).items()}
        self.emission = params.get('emission')
        self.transition = params.get('transition')
        self.tags = params.get('tags')
        self.tag_count = params.get('tag_count')
        self.tag_tag_count = params.get('tag_tag_count')
        self.most_common_tags = params.get('most_common_tags')

    def log_helper(self, word_idx, tag, viterbi, tags1, tags2, tag_flag=True):
        max_val = -np.infty
        max_val_idx = None
        for tag1_idx, tag1 in enumerate(tags1):
            for tag2_idx, tag2 in enumerate(tags2):
                val = viterbi[self.tag_to_idx[tag1]][word_idx - 1] + math.log(self.transition[tag][tag1][tag2])
                if val > max_val:
                    max_val = val
                    if tag_flag:
                        max_val_idx = tag1_idx
                    else:
                        max_val_idx = tag2_idx
        return max_val, max_val_idx

    def calculate_viterbi_log_probability_without_emission(self, word_idx, word, tag, viterbi):
        if word_idx == 1:
            log_transition_prob, most_common_tag_idx = self.log_helper(word_idx, tag, viterbi, self.tags, [START_TAG])
        else:
            log_transition_prob, most_common_tag_idx = self.log_helper(word_idx, tag, viterbi, self.tags, self.tags, False)
        prob = log_transition_prob
        next_id = self.tag_to_idx.get(self.most_common_tags[most_common_tag_idx])
        return prob, next_id

    def helper(self, word_idx, tag, viterbi, tags1, tags2, tag_flag=True):
        max_val = -np.infty
        max_val_idx = None
        for tag1_idx, tag1 in enumerate(tags1):
            for tag2_idx, tag2 in enumerate(tags2):
                val = viterbi[self.tag_to_idx[tag1]][word_idx - 1] * self.transition[tag][tag1][tag2]
                if val > max_val:
                    max_val = val
                    if tag_flag:
                        max_val_idx = tag1_idx
                    else:
                        max_val_idx = tag2_idx
        return max_val, max_val_idx

    def calculate_viterbi_non_log_probability_without_emission(self, word_idx, word, tag, viterbi):
        if word_idx == 1:
            transition_prob, most_common_tag_idx = self.helper(word_idx, tag, viterbi, self.tags, [START_TAG])
        else:
            transition_prob, most_common_tag_idx = self.helper(word_idx, tag, viterbi, self.tags, self.tags, False)
        prob = transition_prob
        next_id = self.tag_to_idx.get(self.most_common_tags[most_common_tag_idx])
        return prob, next_id

    def calculate_viterbi_probability_without_emission(self, word_idx, word, tag, viterbi):
        if self.use_log:
            prob, next_id = self.calculate_viterbi_log_probability_without_emission(word_idx, word, tag, viterbi)
        else:
            prob, next_id = self.calculate_viterbi_non_log_probability_without_emission(word_idx, word, tag, viterbi)
        return prob, next_id

    def calculate_viterbi_log_probability_with_emission(self, word_idx, word, tag, viterbi):
        emission_prob = self.emission.get(word, dict()).get(tag)
        log_emission_prob = math.log(emission_prob) \
            if emission_prob is not None and emission_prob != 0.0 else LARGE_NEGATIVE
        log_transition_prob, next_id = self.log_helper(word_idx, tag, viterbi, self.tags, self.tags, False)
        prob = log_transition_prob + log_emission_prob
        return prob, next_id

    def calculate_viterbi_non_log_probability_with_emission(self, word_idx, word, tag, viterbi):
        emission_prob = self.emission.get(word, dict()).get(tag, 1)
        transition_prob, next_id = self.helper(word_idx, tag, viterbi, self.tags, self.tags, False)
        prob = transition_prob * emission_prob
        return prob, next_id

    def calculate_viterbi_probability_with_emission(self, word_idx, word, tag, viterbi):
        if self.use_log:
            prob, next_id = self.calculate_viterbi_log_probability_with_emission(word_idx, word, tag, viterbi)
        else:
            prob, next_id = self.calculate_viterbi_non_log_probability_with_emission(word_idx, word, tag, viterbi)
        return prob, next_id

    def calculate_initial_viterbi_probability(self, word, viterbi, back_pointer):
        for tag in self.tags:
            tag_idx = self.tag_to_idx.get(tag)
            transition_prob = self.transition.get(tag, dict()).get(START_TAG, dict()).get(START_TAG)
            emission_prob = self.emission.get(word, dict()).get(tag)

            if self.use_log:
                log_transition_prob = math.log(transition_prob)
                log_emission_prob = math.log(emission_prob) \
                    if emission_prob is not None and emission_prob != 0.0 else LARGE_NEGATIVE
                prob = log_transition_prob + log_emission_prob
            else:
                emission_prob = emission_prob if emission_prob is not None else 1.0
                prob = transition_prob * emission_prob
            viterbi[tag_idx][0], back_pointer[tag_idx][0] = prob, 0

    def generate_most_likely_tag_sequence(self, viterbi, back_pointer, seq_len):
        best_tag_idx = int(np.argmax(viterbi[:, seq_len - 1]))
        tag_sequence = []
        for idx in reversed(range(seq_len)):
            tag_sequence.append(self.idx_to_tag[best_tag_idx])
            best_tag_idx = int(back_pointer[best_tag_idx][idx])
        return tag_sequence[::-1]

    def predict(self, data):
        result = OrderedDict()
        for si, words in data.items():
            viterbi = np.zeros((len(self.tags), len(words)))
            back_pointer = np.zeros((len(self.tags), len(words)))
            self.calculate_initial_viterbi_probability(words[0], viterbi, back_pointer)
            for word_idx, word in enumerate(words[1:], start=1):
                for tag_idx, tag in enumerate(self.tags):
                    if self.emission.get(word, dict()).get(tag) is None:
                        prob, next_id = self.calculate_viterbi_probability_without_emission(word_idx, word, tag, viterbi)
                    else:
                        prob, next_id = self.calculate_viterbi_probability_with_emission(word_idx, word, tag, viterbi)

                    viterbi[tag_idx][word_idx] = prob
                    back_pointer[tag_idx][word_idx] = next_id
            tag_sequence = self.generate_most_likely_tag_sequence(viterbi, back_pointer, len(words))
            result[si] = {'words': words, 'tags': tag_sequence}
        return result

    @classmethod
    def load_model(cls, model_path):
        clf = cls()
        with open(model_path) as model:
            model_parameters = json.loads(model.read())
            clf._set_params(model_parameters)
        return clf
