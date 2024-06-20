import argparse
import collections
import json
import math
import os
import re
from collections import defaultdict
from functools import lru_cache
import numpy as np

import sys

sys.path.insert(0, os.path.abspath("./"))
from codebleu import calc_codebleu

import logging

logging.basicConfig(level=logging.ERROR)

# acceptable_threshold = 0.8
bad_case_threshold = 0.3


class BaseTokenizer:
    """A base dummy tokenizer to derive from."""

    def signature(self):
        """
        Returns a signature for the tokenizer.
        :return: signature string
        """
        return "none"

    def __call__(self, line):
        """
        Tokenizes an input line with the tokenizer.
        :param line: a segment to tokenize
        :return: the tokenized line
        """
        return line


class TokenizerRegexp(BaseTokenizer):
    def signature(self):
        return "re"

    def __init__(self):
        self._re = [
            # language-dependent part (assuming Western languages)
            (re.compile(r"([\{-\~\[-\` -\&\(-\+\:-\@\/])"), r" \1 "),
            # tokenize period and comma unless preceded by a digit
            (re.compile(r"([^0-9])([\.,])"), r"\1 \2 "),
            # tokenize period and comma unless followed by a digit
            (re.compile(r"([\.,])([^0-9])"), r" \1 \2"),
            # tokenize dash when preceded by a digit
            (re.compile(r"([0-9])(-)"), r"\1 \2 "),
            # one space only between words
            # NOTE: Doing this in Python (below) is faster
            # (re.compile(r'\s+'), r' '),
        ]

    @lru_cache(maxsize=2 ** 16)
    def __call__(self, line):
        """Common post-processing tokenizer for `13a` and `zh` tokenizers.
        :param line: a segment to tokenize
        :return: the tokenized line
        """
        for (_re, repl) in self._re:
            line = _re.sub(repl, line)

        # no leading or trailing spaces, single space within words
        # return ' '.join(line.split())
        # This line is changed with regards to the original tokenizer (seen above) to return individual words
        return line.split()


class Tokenizer13a(BaseTokenizer):
    def signature(self):
        return "13a"

    def __init__(self):
        self._post_tokenizer = TokenizerRegexp()

    @lru_cache(maxsize=2 ** 16)
    def __call__(self, line):
        """Tokenizes an input line using a relatively minimal tokenization
        that is however equivalent to mteval-v13a, used by WMT.

        :param line: a segment to tokenize
        :return: the tokenized line
        """

        # language-independent part:
        line = line.replace("<skipped>", "")
        line = line.replace("-\n", "")
        line = line.replace("\n", " ")

        if "&" in line:
            line = line.replace("&quot;", '"')
            line = line.replace("&amp;", "&")
            line = line.replace("&lt;", "<")
            line = line.replace("&gt;", ">")

        return self._post_tokenizer(f" {line} ")


def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.

  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.

  Returns:
    The Counter containing all n-grams upto max_order in segment
    with a count of how many times each n-gram occurred.
  """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False):
    """Computes BLEU score of translated segments against one or more references.

  Args:
    reference_corpus: list of lists of references for each translation. Each
        reference should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation
        should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    smooth: Whether or not to apply Lin et al. 2004 smoothing.

  Returns:
    3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
    precisions and brevity penalty.
  """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus,
                                         translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                             (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                 possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length
    # ratio = 1.0
    # ratio = float(reference_length) / translation_length

    if ratio > 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / ratio)

    bleu = geo_mean * bp

    return (bleu, precisions, bp, ratio, translation_length, reference_length)


def calculate_bleu4(predictions, references, tokenizer=Tokenizer13a(), max_order=4, smooth=False):
    # if only one reference is provided make sure we still use list of lists
    if isinstance(references[0], str):
        references = [[ref] for ref in references]

    references = [[tokenizer(r) for r in ref] for ref in references]
    predictions = [tokenizer(p) for p in predictions]

    if len(predictions) == 1 and len(predictions[0]) == 0:
        return 0.0

    reference_split_len = len(references[0][0])
    if reference_split_len == 3:
        max_order = 3
    elif reference_split_len == 2:
        max_order = 2
    elif reference_split_len <= 1:
        max_order = 1
    else:
        max_order = 4

    score = compute_bleu(
        reference_corpus=references, translation_corpus=predictions, max_order=max_order, smooth=smooth
    )
    (bleu, precisions, bp, ratio, translation_length, reference_length) = score
    return bleu
    # return {
    #     "bleu": bleu,
    #     "precisions": precisions,
    #     "brevity_penalty": bp,
    #     "length_ratio": ratio,
    #     "translation_length": translation_length,
    #     "reference_length": reference_length,
    # }


def get_statistics(item_dict):
    statistics = {}
    for task_type, scores in item_dict.items():
        if len(scores) > 0 and isinstance(scores[0], bool):
            scores = np.array(scores).astype(int)
        statistics[task_type] = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            '25th_percentile': np.percentile(scores, 25),
            '50th_percentile': np.percentile(scores, 50),
            '75th_percentile': np.percentile(scores, 75)
        }
    return statistics


def construct_result_str(task_type, exact_match_statistics, acceptable_statistics, bleu4_score_statistics,
                         codebleu_score_statistics,
                         prediction_length_statistics, reference_length_statistics, count_dict):
    em_stats = exact_match_statistics[task_type]
    acp_stats = acceptable_statistics[task_type]
    bl4_stats = bleu4_score_statistics[task_type]
    cb_stats = codebleu_score_statistics[task_type]
    plen_stats = prediction_length_statistics[task_type]
    rlen_stats = reference_length_statistics[task_type]
    count = count_dict[task_type]
    return f"{task_type:<20}\t" \
           f"mean: {em_stats['mean'] * 100:<15.3f}\tmean: {acp_stats['mean'] * 100:<15.3f}\tmean: {bl4_stats['mean']:<15.3f}\tmean: {cb_stats['mean']:<15.3f}\tmean: {plen_stats['mean']:<4.0f} / {rlen_stats['mean']:<10.0f}\t{count:<10}\n\n"
    #    f"mean: {em_stats['mean'] * 100:<15.3f}\tmean: {acp_stats['mean'] * 100:<15.3f}\tmean: {bl4_stats['mean']:<15.3f}\tmean: {cb_stats['mean']:<15.3f}\tmean: {plen_stats['mean']:<4.0f} / {rlen_stats['mean']:<10.0f}\t{count:<10}\n" \
    #    f"{' ':<70}\t std: {bl4_stats['std']:<15.3f}\t std: {cb_stats['std']:<15.3f}\t std: {plen_stats['std']:<4.0f} / {rlen_stats['std']:<10.0f}\n" \
    #    f"{' ':<70}\t25th: {bl4_stats['25th_percentile']:<15.3f}\t25th: {cb_stats['25th_percentile']:<15.3f}\t25th: {plen_stats['25th_percentile']:<4.0f} / {rlen_stats['25th_percentile']:<10.0f}\n" \
    #    f"{' ':<70}\t50th: {bl4_stats['50th_percentile']:<15.3f}\t50th: {cb_stats['50th_percentile']:<15.3f}\t50th: {plen_stats['50th_percentile']:<4.0f} / {rlen_stats['50th_percentile']:<10.0f}\n" \
    #    f"{' ':<70}\t75th: {bl4_stats['75th_percentile']:<15.3f}\t75th: {cb_stats['75th_percentile']:<15.3f}\t75th: {plen_stats['75th_percentile']:<4.0f} / {rlen_stats['75th_percentile']:<10.0f}\n\n"


def score(language, result_jsonl):
    lang = language
    question_jsonl = f'datasets/{lang}_test_8k_full.jsonl'
    answer_jsonl = result_jsonl

    directory_name = os.path.dirname(answer_jsonl)
    base_name = os.path.basename(answer_jsonl)
    file_name, _ = os.path.splitext(base_name)
    scored_file_path = f"{os.path.join(directory_name, file_name + '_scored.jsonl')}"

    with open(question_jsonl, "r") as question_f, \
            open(answer_jsonl, "r") as answer_f, \
            open(scored_file_path, "w") as output_f:
        answer_jsons = [json.loads(line) for line in answer_f.readlines()]
        answer_dict = {j['task_id']: j for j in answer_jsons}

        for line in question_f.readlines():
            question_json = json.loads(line)
            task_id = question_json['task_id']

            prediction = ''
            if task_id in answer_dict:
                answer_json = answer_dict[task_id]
                if 'prediction' in answer_json:
                    prediction = answer_json['prediction'].strip()
                else:
                    print(f'Could not find prediction for task_id {task_id} !')
                    continue
            else:
                print(f'Could not find task_id {task_id} in answer_jsonl {answer_jsonl} file !')
                continue
            canonical_solution = question_json['canonical_solution'].strip()


            exact_match = True if prediction == canonical_solution else False
            # attention:  bleu4_score <- 0.0 if prediction is empty and canonical_solution is not empty
            if exact_match:
                bleu4_score = 1.0
            elif len(prediction) == 0 or len(canonical_solution) == 0:
                bleu4_score = 0.0
            else:
                bleu4_score = calculate_bleu4([prediction], [canonical_solution])

            # codebleu_score = calc_codebleu([canonical_solution], [prediction], lang="java", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)['codebleu']
            if language == 'cplus':
                lang = 'cpp'
            codebleu_score = \
            calc_codebleu([canonical_solution], [prediction], lang=lang, weights=(0.25, 0.25, 0.25, 0.25),
                          tokenizer=Tokenizer13a())['codebleu']

            scored_json = {'task_id': task_id, 'task_type': question_json['task_type'],
                           'canonical_solution': question_json['canonical_solution'], 'prediction': prediction,
                           "exact_match": exact_match, "bleu4_score": bleu4_score, "codebleu_score": codebleu_score}

            output_f.write(json.dumps(scored_json, ensure_ascii=False) + "\n")
            output_f.flush()

    return scored_file_path


def analyze(scored_file_path, result_jsonl):
    answer_jsonl = result_jsonl
    directory_name = os.path.dirname(answer_jsonl)
    base_name = os.path.basename(answer_jsonl)
    file_name, _ = os.path.splitext(base_name)

    with open(scored_file_path, "r") as scored_f, \
            open(f"{os.path.join(directory_name, file_name + '_statistics.txt')}", "w") as output_f:
        input_jsons = [json.loads(line) for line in scored_f.readlines()]

        exact_match_dict = defaultdict(list)
        # acceptable_dict = defaultdict(list)
        bleu4_score_dict = defaultdict(list)
        codebleu_score_dict = defaultdict(list)
        prediction_length_dict = defaultdict(list)
        reference_length_dict = defaultdict(list)
        count_dict = defaultdict(int)

        for input_json in input_jsons:
            if 'task_type' not in input_json:
                input_json['input_json'] = 'default'

            # attention
            # if input_json['task_type'] in ['multiple_lines', 'random']:
            #     continue

            exact_match_dict[input_json['task_type']].append(input_json['exact_match'])
            exact_match_dict['Total'].append(input_json['exact_match'])

            # acceptable = True if input_json['bleu4_score'] >= acceptable_threshold else False
            # acceptable_dict[input_json['task_type']].append(acceptable)
            # acceptable_dict['Total'].append(acceptable)

            bleu4_score_dict[input_json['task_type']].append(input_json['bleu4_score'])
            bleu4_score_dict['Total'].append(input_json['bleu4_score'])

            codebleu_score_dict[input_json['task_type']].append(input_json['codebleu_score'])
            codebleu_score_dict['Total'].append(input_json['codebleu_score'])

            prediction_length = len(input_json['prediction'])
            prediction_length_dict[input_json['task_type']].append(prediction_length)
            prediction_length_dict['Total'].append(prediction_length)

            reference_length = len(input_json['canonical_solution'])
            reference_length_dict[input_json['task_type']].append(reference_length)
            reference_length_dict['Total'].append(reference_length)

            count_dict[input_json['task_type']] += 1
            count_dict['Total'] += 1

        exact_match_statistics = get_statistics(exact_match_dict)
        # acceptable_statistics = get_statistics(acceptable_dict)
        bleu4_score_statistics = get_statistics(bleu4_score_dict)
        codebleu_score_statistics = get_statistics(codebleu_score_dict)
        prediction_length_statistics = get_statistics(prediction_length_dict)
        reference_length_statistics = get_statistics(reference_length_dict)

        result_str = f"{'Task Type':<20}\t{'Exact Match %':<20}\t{'BLEU-4':<20}\t{'CODE-BLEU':<20}\t{'Length(Pred/Ref)':<20}\t{'Case Number':<20}\n"

        task_types = [key for key in exact_match_statistics.keys() if key != 'Total']
        for task_type in task_types:
            result_str += construct_result_str(task_type, exact_match_statistics, 
                                               bleu4_score_statistics, codebleu_score_statistics,
                                               prediction_length_statistics,
                                               reference_length_statistics, count_dict)
        result_total = construct_result_str('Total', exact_match_statistics,
                                           bleu4_score_statistics, codebleu_score_statistics,
                                           prediction_length_statistics,
                                           reference_length_statistics, count_dict)
        result_str += result_total

        result_str += f"{'Task Type':<20}\t{'Exact Match %':<20}\t{'BLEU-4':<20}\t{'CODE-BLEU':<20}\t{'Length(Pred/Ref)':<20}\t{'Case Number':<20}\n"
  
        print(result_str)

        output_f.write(result_str)
        output_f.flush()
        return result_total


def main(config):
    # print(config)
    # print(config.language)
    # print(config.result_path)
    languages = config.language.split(' ')
    result_dir = config.result_path.split('/')[-1]
    result_total = "=====" * 10 + "\n"
    result_total += f"{'Task Type':<20}\t{'Exact Match %':<20}\t{'BLEU-4':<20}\t{'CODE-BLEU':<20}\t{'Length(Pred/Ref)':<20}\t{'Case Number':<20}\n"
    for language in languages:
        if language == 'cpp':
            language == 'cplus'
        # python_aiXcoder-7b-base-weights_12k.jsonl
        result_jsonl = f'{config.result_path}/{language}_{result_dir}_8k.jsonl'
        if os.path.exists(result_jsonl):
            print(f'{language} is evaluating...')
            scored_file_path = score(language, result_jsonl)
            result = analyze(scored_file_path, result_jsonl)
            result_total += f'language：{language}\n{result}'
        else:
            print(f'{language} result file is not exists')
    print(result_total)


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Result Analyzer.')

    # 设置评测的语言
    parser.add_argument(
        "--language",
        type=str,
        default="python java cpp javascript",
        help="Language of Dataset to evaluate",
    )
    # 设置数据集文件
    # parser.add_argument(
    #     "--full_jsonl",
    #     type=str,
    #     default="./datasets",
    #     help="Path to which the Codex responses will be cached at",
    # )
    # 设置预测结果文件
    parser.add_argument(
        "--result_path",
        type=str,
        default="outputs/aiXcoder-7b-base-weights",
        help="Path to which the Evaluate results will be cached at",
    )
    # Parse the arguments
    args = parser.parse_args()

    main(args)
