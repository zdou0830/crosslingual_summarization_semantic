# -*- coding: utf-8 -*-


def get_ngrams(n, text):
    ngram_set = set()
    text_length = len(text)
    max_index_ngram = text_length - n
    for i in range(max_index_ngram + 1):
        ngram_set.add(tuple(text[i:i+n]))
    return ngram_set
import collections
def create_ngrams(tokens, n):
    ngrams= collections.Counter()
    for ngram in (tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)):
      ngrams[ngram] += 1
    return ngrams

import six

def score_ngrams(target_ngrams, prediction_ngrams):
    intersection_ngrams_count = 0
    for ngram in six.iterkeys(target_ngrams):
      intersection_ngrams_count += min(target_ngrams[ngram],
                                       prediction_ngrams[ngram])
    target_ngrams_count = sum(target_ngrams.values())
    prediction_ngrams_count = sum(prediction_ngrams.values())

    prec = intersection_ngrams_count / max(prediction_ngrams_count, 1)
    rec = intersection_ngrams_count / max(target_ngrams_count, 1)

    return 2*prec*rec / (rec+prec+1e-12)

def rouge_n(evaluated_sentences, reference_sentences, n=2):  #默认rouge_2
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        return 0


    evaluated_ngrams = get_ngrams(n, evaluated_sentences)
    reference_ngrams = get_ngrams(n, reference_sentences)
    reference_ngrams_count = len(reference_ngrams)
    if reference_ngrams_count == 0:
        return 0

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_ngrams_count = len(overlapping_ngrams)
    return overlapping_ngrams_count / reference_ngrams_count


def rouge_1(evaluated_sentences, reference_sentences):
    evaluated_sentences = evaluated_sentences.split()
    reference_sentences = reference_sentences.split()
    return rouge_n(evaluated_sentences, reference_sentences, n=1)


def rouge_2(evaluated_sentences, reference_sentences):
    evaluated_sentences = evaluated_sentences.split()
    reference_sentences = reference_sentences.split()
    return rouge_n(evaluated_sentences, reference_sentences, n=2)


def F_1(evaluated_sentences, reference_sentences, beta=1):
    evaluated_sentences = evaluated_sentences.split()
    reference_sentences = reference_sentences.split()
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        return 0
    target_ngrams = create_ngrams(reference_sentences, beta)
    prediction_ngrams = create_ngrams(evaluated_sentences, beta)
    scores = score_ngrams(target_ngrams, prediction_ngrams)
    return scores

    evaluated_ngrams = get_ngrams(beta, evaluated_sentences)  # equal to retrieved set
    reference_ngrams = get_ngrams(beta, reference_sentences)  # equal to relevant set
    evaluated_ngrams_num = len(evaluated_ngrams)
    reference_ngrams_num = len(reference_ngrams)

    if reference_ngrams_num == 0 or evaluated_ngrams_num == 0:
        return 0

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_ngrams_num = len(overlapping_ngrams)
    if overlapping_ngrams_num == 0:
        return 0
    return 2*overlapping_ngrams_num / (reference_ngrams_num + evaluated_ngrams_num)

def my_lcs(string, sub):
    if(len(string)< len(sub)):
        sub, string = string, sub

    lengths = [[0 for i in range(0,len(sub)+1)] for j in range(0,len(string)+1)]

    for j in range(1,len(sub)+1):
        for i in range(1,len(string)+1):
            if(string[i-1] == sub[j-1]):
                lengths[i][j] = lengths[i-1][j-1] + 1
            else:
                lengths[i][j] = max(lengths[i-1][j] , lengths[i][j-1])

    return lengths[len(string)][len(sub)]
def calc_rouge_L(candidate, reference):
    """
    Compute ROUGE-L score given one candidate and references for an image
    :param candidate: str : candidate sentence to be evaluated
    :param refs: list of str : COCO reference sentences for the particular image to be evaluated
    :returns score: int (ROUGE-L score for the candidate evaluated against references)
    """
    beta = 1.
    prec = []
    rec = []
    # split into tokens
    token_c = candidate#.split(" ")
    token_r = reference#.split(" ")
    # compute the longest common subsequence
    lcs = my_lcs(token_r, token_c)
    prec_max = (lcs/float(len(token_c)))
    rec_max = (lcs/float(len(token_r)))

    if(prec_max!=0 and rec_max !=0):
        score = ((1 + beta**2)*prec_max*rec_max)/float(rec_max + beta**2*prec_max)
    else:
        score = 0.0
    return score
