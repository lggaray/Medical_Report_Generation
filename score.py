import argparse
import os
import similarity
import pickle

from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu


def get_args():
    parser = argparse.ArgumentParser(description='Command-line script for BLEU scoring.')
    parser.add_argument('--system', required=True, help='system output')
    parser.add_argument('--reference', required=True, help='references')
    parser.add_argument('--orders', nargs='+', type=int, default=[1, 2, 3, 4], help='n-grams for calculating BLEU scores')
    parser.add_argument('--ignore-case', action='store_true', help='case-insensitive scoring')
    return parser.parse_args()


def readlines(filename, ignore_case=False, wrapped=False):
    lines = []
    with open(filename) as file:
        for line in file.readlines():
            line = line.rstrip()
            if ignore_case:
                line = line.lower()
            line = word_tokenize(line)
            lines.append([line] if wrapped else line)
    return lines


def main(args):
    assert os.path.exists(args.system), "System output file {} does not exist".format(args.system)
    assert os.path.exists(args.reference), "Reference file {} does not exist".format(args.reference)

    scores = {}
    reference = readlines(args.reference, wrapped=True)
    system = readlines(args.system)
    cc = SmoothingFunction()
    dic = {'emb': [], 'bow': [], 'bleu': []}
    emb, bow, n = 0, 0, 0
    for i in range(len(reference)):
        aux1, aux2 = similarity.sim(system[i], reference[i][0])
        emb += aux1
        bow += aux2
        dic['emb'].append(aux1)
        dic['bow'].append(aux2)
        n += 1
    for order in args.orders:
        accum = 0
        for i in range(len(reference)):
            try:
                aux = sentence_bleu(reference[i], system[i], weights=(1.0 / order,) * order,  smoothing_function=cc.method4)
            except:
                aux = 0
            if order == 1:
                dic['bleu'].append(aux)
            accum += aux
        scores[order] = accum / len(reference)
    print(', '.join('BLEU{} = {:.4f}'.format(order, 100 * score) for order, score in scores.items()))
    print('Coseno con emb: {}, Coseno con bow: {}'.format(emb/n, bow/n))
    with open("scores.p", "wb") as f:
        pickle.dump(dic, f)

if __name__ == '__main__':
    args = get_args()
    main(args)
