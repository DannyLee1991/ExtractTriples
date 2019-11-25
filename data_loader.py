import json
import os
import numpy as np
import re
from random import choice
from tqdm import tqdm
import ahocorasick
from config import *

import pyhanlp


def tokenize(s):
    return [i.word for i in pyhanlp.HanLP.segment(s)]


def make_random_order_vote(total_data, target_file=random_order_vote_path):
    if not os.path.exists(target_file):
        random_order = [i for i in range(len(total_data))]
        np.random.shuffle(random_order)
        json.dump(
            random_order,
            open(target_file, 'w'),
            indent=4
        )
    else:
        random_order = json.load(open(target_file))
    return random_order


def repair(d):
    d['text'] = d['text'].lower()
    something = re.findall(u'《([^《》]*?)》', d['text'])
    something = [s.strip() for s in something]
    zhuanji = []
    gequ = []
    for sp in d['spo_list']:
        sp[0] = sp[0].strip(u'《》').strip().lower()
        sp[2] = sp[2].strip(u'《》').strip().lower()
        for some in something:
            if sp[0] in some and d['text'].count(sp[0]) == 1:
                sp[0] = some
        if sp[1] == '所属专辑':
            zhuanji.append(sp[2])
            gequ.append(sp[0])
    spo_list = []
    for sp in d['spo_list']:
        if sp[1] in ['歌手', '作词', '作曲']:
            if sp[0] in zhuanji and sp[0] not in gequ:
                continue
        spo_list.append(tuple(sp))
    d['spo_list'] = spo_list


def random_generate(d, spo_list_key):
    r = np.random.random()
    if r > 0.5:
        return d
    else:
        k = np.random.randint(len(d[spo_list_key]))
        spi = d[spo_list_key][k]
        k = np.random.randint(len(predicates[spi[1]]))
        spo = predicates[spi[1]][k]
        F = lambda s: s.replace(spi[0], spo[0]).replace(spi[2], spo[2])
        text = F(d['text'])
        spo_list = [(F(sp[0]), sp[1], F(sp[2])) for sp in d[spo_list_key]]
        return {'text': text, spo_list_key: spo_list}


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


def sent2vec(S):
    """S格式：[[w1, w2]]
    """
    from w2v_model import word2vec, word2id
    V = []
    for s in S:
        V.append([])
        for w in s:
            for _ in w:
                V[-1].append(word2id.get(w, 0))
    V = seq_padding(V)
    V = word2vec[V]
    return V


def extract_items(text_in, spoer, subject_model, object_model):
    text_words = tokenize(text_in.lower())
    text_in = ''.join(text_words)
    pre_items = {}
    for sp in spoer.extract_items(text_in):
        subjectid = text_in.find(sp[0])
        objectid = text_in.find(sp[2])
        if subjectid != -1 and objectid != -1:
            key = (subjectid, subjectid + len(sp[0]))
            if key not in pre_items:
                pre_items[key] = []
            pre_items[key].append((objectid,
                                   objectid + len(sp[2]),
                                   predicate2id[sp[1]]))
    _pres = np.zeros((len(text_in), 2))
    for j in pre_items:
        _pres[j[0], 0] = 1
        _pres[j[1] - 1, 1] = 1
    _pres = np.expand_dims(_pres, 0)
    R = []
    _t1 = [char2id.get(c, 1) for c in text_in]
    _t1 = np.array([_t1])
    _t2 = sent2vec([text_words])
    _k1, _k2 = subject_model.predict([_t1, _t2, _pres])
    _k1, _k2 = _k1[0, :, 0], _k2[0, :, 0]
    _k1, _k2 = np.where(_k1 > 0.5)[0], np.where(_k2 > 0.4)[0]
    _subjects, _PREO = [], []
    for i in _k1:
        j = _k2[_k2 >= i]
        if len(j) > 0:
            j = j[0]
            _subject = text_in[i: j + 1]
            _subjects.append((_subject, i, j))
            _preo = np.zeros((len(text_in), num_classes, 2))
            for _ in pre_items.get((i, j + 1), []):
                _preo[_[0], _[2], 0] = 1
                _preo[_[1] - 1, _[2], 1] = 1
            _preo = _preo.reshape((len(text_in), -1))
            _PREO.append(_preo)
    if _subjects:
        _PRES = np.repeat(_pres, len(_subjects), 0)
        _PREO = np.array(_PREO)
        _t1 = np.repeat(_t1, len(_subjects), 0)
        _t2 = np.repeat(_t2, len(_subjects), 0)
        _k1, _k2 = np.array([_s[1:] for _s in _subjects]).T.reshape((2, -1, 1))
        _o1, _o2 = object_model.predict([_t1, _t2, _k1, _k2, _PRES, _PREO])
        for i, _subject in enumerate(_subjects):
            _oo1, _oo2 = np.where(_o1[i] > 0.5), np.where(_o2[i] > 0.4)
            for _ooo1, _c1 in zip(*_oo1):
                for _ooo2, _c2 in zip(*_oo2):
                    if _ooo1 <= _ooo2 and _c1 == _c2:
                        _object = text_in[_ooo1: _ooo2 + 1]
                        _predicate = id2predicate[_c1]
                        R.append((_subject[0], _predicate, _object))
                        break
        zhuanji, gequ = [], []
        for s, p, o in R[:]:
            if p == u'妻子':
                R.append((o, u'丈夫', s))
            elif p == u'丈夫':
                R.append((o, u'妻子', s))
            if p == u'所属专辑':
                zhuanji.append(o)
                gequ.append(s)
        spo_list = set()
        for s, p, o in R:
            if p in [u'歌手', u'作词', u'作曲']:
                if s in zhuanji and s not in gequ:
                    continue
            spo_list.add((s, p, o))
        return list(spo_list)
    else:
        return []


class DataGenerator:
    def __init__(self, data, batch_size=64):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = [i for i in range(len(self.data))]
            np.random.shuffle(idxs)
            T1, T2, S1, S2, K1, K2, O1, O2, PRES, PREO = [], [], [], [], [], [], [], [], [], []
            for i in idxs:
                spo_list_key = 'spo_list'  # if np.random.random() > 0.5 else 'spo_list_with_pred'
                d = random_generate(self.data[i], spo_list_key)
                text = d['text'][:maxlen]
                text_words = tokenize(text)
                text = ''.join(text_words)
                items = {}
                for sp in d[spo_list_key]:
                    subjectid = text.find(sp[0])
                    objectid = text.find(sp[2])
                    if subjectid != -1 and objectid != -1:
                        key = (subjectid, subjectid + len(sp[0]))
                        if key not in items:
                            items[key] = []
                        items[key].append((objectid,
                                           objectid + len(sp[2]),
                                           predicate2id[sp[1]]))
                pre_items = {}
                for sp in spoer.extract_items(text, i):
                    subjectid = text.find(sp[0])
                    objectid = text.find(sp[2])
                    if subjectid != -1 and objectid != -1:
                        key = (subjectid, subjectid + len(sp[0]))
                        if key not in pre_items:
                            pre_items[key] = []
                        pre_items[key].append((objectid,
                                               objectid + len(sp[2]),
                                               predicate2id[sp[1]]))
                if items:
                    T1.append([char2id.get(c, 1) for c in text])  # 1是unk，0是padding
                    T2.append(text_words)
                    s1, s2 = np.zeros(len(text)), np.zeros(len(text))
                    for j in items:
                        s1[j[0]] = 1
                        s2[j[1] - 1] = 1
                    pres = np.zeros((len(text), 2))
                    for j in pre_items:
                        pres[j[0], 0] = 1
                        pres[j[1] - 1, 1] = 1
                    k1, k2 = np.array(sorted(items.keys())).T
                    k1 = choice(k1)
                    k2 = choice(k2[k2 >= k1])
                    o1, o2 = np.zeros((len(text), num_classes)), np.zeros((len(text), num_classes))
                    for j in items.get((k1, k2), []):
                        o1[j[0], j[2]] = 1
                        o2[j[1] - 1, j[2]] = 1
                    preo = np.zeros((len(text), num_classes, 2))
                    for j in pre_items.get((k1, k2), []):
                        preo[j[0], j[2], 0] = 1
                        preo[j[1] - 1, j[2], 1] = 1
                    preo = preo.reshape((len(text), -1))
                    S1.append(s1)
                    S2.append(s2)
                    K1.append([k1])
                    K2.append([k2 - 1])
                    O1.append(o1)
                    O2.append(o2)
                    PRES.append(pres)
                    PREO.append(preo)
                    if len(T1) == self.batch_size or i == idxs[-1]:
                        T1 = seq_padding(T1)
                        T2 = sent2vec(T2)
                        S1 = seq_padding(S1)
                        S2 = seq_padding(S2)
                        O1 = seq_padding(O1, np.zeros(num_classes))
                        O2 = seq_padding(O2, np.zeros(num_classes))
                        K1, K2 = np.array(K1), np.array(K2)
                        PRES = seq_padding(PRES, np.zeros(2))
                        PREO = seq_padding(PREO, np.zeros(num_classes * 2))
                        yield [T1, T2, S1, S2, K1, K2, O1, O2, PRES, PREO], None
                        T1, T2, S1, S2, K1, K2, O1, O2, PRES, PREO = [], [], [], [], [], [], [], [], [], []


class SpoSearcher:
    def __init__(self, train_data):
        self.s_ac = ahocorasick.Automaton()
        self.o_ac = ahocorasick.Automaton()
        self.sp2o = {}
        self.spo_total = {}
        for i, d in tqdm(enumerate(train_data), desc=u'构建三元组搜索器'):
            for s, p, o in d['spo_list']:
                self.s_ac.add_word(s, s)
                self.o_ac.add_word(o, o)
                if (s, o) not in self.sp2o:
                    self.sp2o[(s, o)] = set()
                if (s, p, o) not in self.spo_total:
                    self.spo_total[(s, p, o)] = set()
                self.sp2o[(s, o)].add(p)
                self.spo_total[(s, p, o)].add(i)
        self.s_ac.make_automaton()
        self.o_ac.make_automaton()

    def extract_items(self, text_in, text_idx=None):
        R = set()
        for s in self.s_ac.iter(text_in):
            for o in self.o_ac.iter(text_in):
                if (s[1], o[1]) in self.sp2o:
                    for p in self.sp2o[(s[1], o[1])]:
                        if text_idx is None:
                            R.add((s[1], p, o[1]))
                        elif (self.spo_total[(s[1], p, o[1])] - set([text_idx])):
                            R.add((s[1], p, o[1]))
        return list(R)


# -------------------------------------------------------------
# -------------------------------------------------------------


total_data = json.load(open(train_data_me_path))
id2predicate, predicate2id = json.load(open(all_50_schemas_me_path))
id2predicate = {int(i): j for i, j in id2predicate.items()}
id2char, char2id = json.load(open(all_chars_me_path))
num_classes = len(id2predicate)

if not os.path.exists(random_order_vote_path):
    random_order = [i for i in range(len(total_data))]
    np.random.shuffle(random_order)
    json.dump(
        random_order,
        open(random_order_vote_path, 'w'),
        indent=4
    )
else:
    random_order = json.load(open(random_order_vote_path))

train_data = [total_data[j] for i, j in enumerate(random_order) if i % 8 != mode]
dev_data = [total_data[j] for i, j in enumerate(random_order) if i % 8 == mode]

predicates = {}

for d in train_data:
    repair(d)
    for sp in d['spo_list']:
        if sp[1] not in predicates:
            predicates[sp[1]] = []
        predicates[sp[1]].append(sp)

for d in dev_data:
    repair(d)
