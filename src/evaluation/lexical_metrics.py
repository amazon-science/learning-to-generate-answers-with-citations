import logging
import re
import string
from collections import Counter
from statistics import mean

import numpy as np
import sacrebleu
import spacy
from nltk import sent_tokenize, word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.util import ngrams
from rouge_score import rouge_scorer, scoring
from sacrebleu.metrics import BLEU, CHRF
from sklearn.metrics import accuracy_score

from src.utils.util import normalize_answer, remove_citations

logger = logging.getLogger(__name__)


def clean_text_faithdial(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the|in|our)\b", " ", text)
    return re.sub(" +", " ", text).strip()


def compute_qampari_f1(answers, generations):
    prec = []
    rec = []
    rec_top5 = []
    f1 = []
    f1_top5 = []

    num_preds = []
    for i, answer in enumerate(answers):
        o = generations[i]
        preds = [
            normalize_answer(x.strip())
            for x in o.rstrip().rstrip(".").rstrip(",").rstrip(QAMPARI_SEP_TOKEN).split(QAMPARI_SEP_TOKEN)
        ]
        preds = [p for p in preds if len(p) > 0]  # delete empty answers
        num_preds.append(len(preds))
        answers = [[normalize_answer(x) for x in ans] for ans in answer]
        flat_answers = [item for sublist in answers for item in sublist]

        prec.append(sum([p in flat_answers for p in preds]) / len(preds) if len(preds) > 0 else 0)
        rec.append(sum([any([x in preds for x in a]) for a in answers]) / len(answers))
        rec_top5.append(min(5, sum([any([x in preds for x in a]) for a in answers])) / min(5, len(answers)))
        if (prec[-1] + rec[-1]) == 0:
            f1.append(0)
        else:
            f1.append(2 * prec[-1] * rec[-1] / (prec[-1] + rec[-1]))
        if (prec[-1] + rec_top5[-1]) == 0:
            f1_top5.append(0)
        else:
            f1_top5.append(2 * prec[-1] * rec_top5[-1] / (prec[-1] + rec_top5[-1]))

    return [
        {
            "num_preds": np.mean(num_preds),
            "qampari_prec": 100 * np.mean(prec),
            "qampari_rec": 100 * np.mean(rec),
            "qampari_rec_top5": 100 * np.mean(rec_top5),
            "qampari_f1": 100 * np.mean(f1),
            "qampari_f1_top5": 100 * np.mean(f1_top5),
        },
        {
            "qampari_prec_indiv": prec,
            "qampari_rec_indiv": rec,
            "qampari_rec_top5_indiv": rec_top5,
            "qampari_f1_indiv": f1,
            "qampari_f1_top5_indiv": f1_top5,
        },
    ]


def compute_str_em_normalized(qa_pairs, generations, attributed_sentences):
    """Compute STR-EM metric (only for ASQA)
    Args:
        data: requires field `qa_pairs/short_answers` and `output`
    Returns:
        STR-EM and STR-EM-HIT ()
    """

    def exact_presence(short_answers, context):
        """Verify if any of the answers is present in the given context.
        Args:
            short_answers: list of short answers to look for in the context
            context: a paragraph to search for short answers
        Returns:
            true if any of the short answers is present in the context
        """

        n_short_answers = [normalize_answer(sa) for sa in short_answers]
        n_context = normalize_answer(context)

        for ans in n_short_answers:
            if ans in n_context:
                return True

        return False

    if qa_pairs is None:
        return 0, 0

    acc = []
    hit = []

    generations_normalized = []
    for i, generation in enumerate(generations):
        generation = remove_citations(generation)
        generation = sent_tokenize(generation)
        generation = "".join([x for j, x in enumerate(generation) if j in attributed_sentences[i]])
        generations_normalized.append(generation)

    generations = generations_normalized

    for i, generation in enumerate(generations):
        loc_acc = []
        for qa_pair in qa_pairs[i]:
            loc_acc.append(exact_presence(qa_pair["short_answers"], generation))
        acc.append(np.mean(loc_acc))
        hit.append(int(np.mean(loc_acc) == 1))

    return 100 * np.mean(acc), 100 * np.mean(hit), acc, hit


def compute_str_em(qa_pairs, generations, attributed_sentences=None):
    """Compute STR-EM metric (only for ASQA)
    Args:
        data: requires field `qa_pairs/short_answers` and `output`
    Returns:
        STR-EM and STR-EM-HIT ()
    """

    def exact_presence(short_answers, context):
        """Verify if any of the answers is present in the given context.
        Args:
            short_answers: list of short answers to look for in the context
            context: a paragraph to search for short answers
        Returns:
            true if any of the short answers is present in the context
        """

        n_short_answers = [normalize_answer(sa) for sa in short_answers]
        n_context = normalize_answer(context)

        # print("str_em eval")
        # print(n_short_answers)
        # print(n_context)
        # print("------")
        for ans in n_short_answers:
            if ans in n_context:
                return True

        return False

    if qa_pairs is None:
        return 0, 0

    acc = []
    hit = []

    if attributed_sentences:
        generations_normalized = []
        for i, generation in enumerate(generations):
            generation = sent_tokenize(generations)
            generation = "".join([x for x in generations if x in attributed_sentences[i]])
            generations_normalized.append(generation)
        generations = generations_normalized
    for i, generation in enumerate(generations):
        loc_acc = []
        for qa_pair in qa_pairs[i]:
            loc_acc.append(exact_presence(qa_pair["short_answers"], generation))
        acc.append(np.mean(loc_acc))
        hit.append(int(np.mean(loc_acc) == 1))

    return 100 * np.mean(acc), 100 * np.mean(hit), acc, hit


def compute_rouge(hypotheses, references1, references2):
    """Main function for rouge scoring.
    If two references are provided,
    the best score is chosen for each instance.
    Args:
        data: requires field `output` and `answer` (or `annotations` for ASQA)
        metrics: list of evaluation metrics
    Returns:
        dictionary representation of rouge scores
    """

    def _rouge_calculation(hypotheses, references1, references2=[], metrics=["rougeLsum"]):

        if references2 == []:
            references2 = references1

        scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
        aggregator = scoring.BootstrapAggregator()

        for i in range(len(hypotheses)):
            scores1 = scorer.score(references1[i], hypotheses[i])
            scores2 = scorer.score(references2[i], hypotheses[i])
            if scores1["rougeLsum"].fmeasure > scores2["rougeLsum"].fmeasure:
                aggregator.add_scores(scores1)
            else:
                aggregator.add_scores(scores2)

        scores = {m: [] for m in metrics}

        for m in metrics:
            fmeasure = aggregator.aggregate()[m].mid.fmeasure
            scores[m].append(fmeasure)

        for m in scores:
            scores[m] = 100 * sum(scores[m]) / len(scores[m])

        return scores

    h, r1, r2 = [], [], []

    for i, reference1 in enumerate(references1):
        h.append(hypotheses[i])
        r1.append(reference1)

        if references2 is not None:
            r2.append(references2[i])

    h = ["\n".join(sent_tokenize(text.lower())) for text in h]
    r1 = ["\n".join(sent_tokenize(text.lower())) for text in r1]
    r2 = ["\n".join(sent_tokenize(text.lower())) for text in r2]
    scores = _rouge_calculation(h, r1, r2)

    return scores["rougeLsum"]