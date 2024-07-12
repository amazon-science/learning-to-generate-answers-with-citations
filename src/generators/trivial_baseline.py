import json
import os

from nltk import sent_tokenize


def truncate_answer(answer, length, tokenizer):
    sents = sent_tokenize(answer)
    curr_ans = []
    curr_len = 0
    for sent in sents:
        len_sent = len(tokenizer(sent)["input_ids"])
        if curr_len + len_sent > length and curr_len != 0:  # ensure that at least one sentence is contained in answer
            break
        curr_len += len_sent
        curr_ans.append(sent)

    curr_ans = " ".join(curr_ans)
    return curr_ans


def trivial_reader_baseline(passages, max_length, tokenizer):
    curr_ans = []
    for i, passage in enumerate(passages):
        sents = sent_tokenize(passage["text"])
        for (
            sent
        ) in (
            sents
        ):  # Either include or exclude the last sentence, which is normally incomplete and results in lower scores.
            curr_ans.append("[{}] {}".format(str(i + 1), passage["title"] + ": " + sent))  # TODO: passage["title"]
    answer = " ".join(curr_ans)
    tokens = tokenizer(answer)["input_ids"]
    answer = tokens[
        :max_length
    ]  # Assuming that word level tokens are on average 20 longer than for neural counterpart
    answer = tokenizer.decode(answer)
    answer = truncate_answer(answer, max_length, tokenizer)
    return answer


def run_trivial_reader_baseline(retrieved_passages, max_length, tokenizer):
    generations = []
    for i, passages in enumerate(retrieved_passages):
        generation = trivial_reader_baseline(passages, max_length, tokenizer)
        generations.append(generation)
    return generations
