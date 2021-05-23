from gensim.models import FastText
from gensim.test.utils import common_texts
import json
import os
import pickle
from typing import List
# print(common_texts[0])
# min_count = 1
# extra_words = [['<PAD>' for _ in range(min_count)] + ['<START>' for _ in range(min_count)] +\
#                    ['<EOS>' for _ in range(min_count)] + ['<UNK>' for _ in range(min_count)]]
# model = FastText(size=128, window=5, min_count=min_count)  # instantiate
# model.build_vocab(sentences=extra_words)
# model.build_vocab(sentences=common_texts, update=True)
# model.train(sentences=common_texts, total_examples=len(common_texts), epochs=10)
# print(model.wv['human'])

def is_alpha(string):
    for x in string:
        if 64 < ord(x) and ord(x) < 123:
            continue
        else:
            return False
    return True

def json2corpus(file_path):
    res = []
    with open(file_path,'r', encoding='utf8') as f:
        for method_json in f.readlines():
            try:
                fst_part, sec_part = method_json.split(', ')
                token, test_name = json.loads(fst_part), json.loads(sec_part)
                # method_json = json.loads(method_json)
            except:
                continue
            # content = method_json[0]
            # name = method_json[1]
            cur_line = token + test_name

            # cur_line = method_json[-1]
            # # cur_line = method_json[-1] + method_json[-2]  # only for kuis dataset
            # for x in method_json[:-1]:
            # # for x in method_json[:-2]:  # only for kuis dataset
            #     if is_alpha(x[0]):
            #         cur_line.append(x[0])
            res.append(cur_line)
    return res


def save_vocab_weight(model, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, 'vocab.pkl'), 'wb') as f_vocab, open(os.path.join(output_dir,'weight.pkl'),'wb') as f_weight:
            pickle.dump(model.wv.index2word, f_vocab)
            pickle.dump(model.wv.vectors_vocab[:len(model.wv.index2word), :], f_weight)

def main(corpus_path:List[str], output_dir):
    extra_words = [['<PAD>' for _ in range(min_count)] + ['<START>' for _ in range(min_count)] +\
                   ['<EOS>' for _ in range(min_count)] + ['<UNK>' for _ in range(min_count)]]
    model = FastText(size=v_dim, window=window, min_count=min_count)  # instantiate
    model.build_vocab(sentences=extra_words)
    corpus = []
    for cp in corpus_path:
        corpus += json2corpus(cp)
    model.build_vocab(sentences=corpus, update=True)
    # model.train(sentences=corpus, total_examples=len(corpus), epochs=5)
    save_vocab_weight(model, output_dir)

if __name__ == '__main__':
    v_dim = 128
    window = 8
    min_count = 1
    main(['train.json',
          'validation.json',
          'test.json',
          ], './dataset/fasttext_vectors')

