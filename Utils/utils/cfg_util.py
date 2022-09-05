'''
Author: 成凯阳
Date: 2022-03-23 13:15:02
LastEditors: 成凯阳
LastEditTime: 2022-03-23 14:02:43
FilePath: /Main/Utils/utils/cfg_util.py

Copyright (c) 2022 by 用户/公司名, All Rights Reserved. 
'''
import nltk

import numpy as np
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


import numpy as np

from Utils.utils import zinc_grammar


def get_zinc_tokenizer(cfg):
    long_tokens = [a for a in cfg._lexical_index.keys() if len(a) > 1]
    replacements = ['$', '%', '^']
    assert len(long_tokens) == len(replacements)
    for token in replacements:
        assert token not in cfg._lexical_index

    def tokenize(smiles):
        for i, token in enumerate(long_tokens):
            smiles = smiles.replace(token, replacements[i])
        tokens = []
        for token in smiles:
            try:
                ix = replacements.index(token)
                tokens.append(long_tokens[ix])
            except Exception:
                tokens.append(token)
        return tokens
    return tokenize


def encode(smiles):
    GCFG = zinc_grammar.GCFG
    tokenize = get_zinc_tokenizer(GCFG)
    tokens = tokenize(smiles)
    parser = nltk.ChartParser(GCFG)
    # if you use Python 2: parse_tree = parser.parse(tokens).next()
    parse_tree = parser.parse(tokens).__next__()
    productions_seq = parse_tree.productions()
    productions = GCFG.productions()
    prod_map = {}
    for ix, prod in enumerate(productions):
        prod_map[prod] = ix
    indices = np.array([prod_map[prod] for prod in productions_seq], dtype=int)
    return indices


def prods_to_eq(prods):
    seq = [prods[0].lhs()]
    for prod in prods:
        if str(prod.lhs()) == 'Nothing':
            break
        for ix, s in enumerate(seq):
            if s == prod.lhs():
                seq = seq[:ix] + list(prod.rhs()) + seq[ix+1:]
                break
    try:
        return ''.join(seq)
    except Exception:
        return ''


def decode(rule):
    productions = zinc_grammar.GCFG.productions()
    prod_seq = [productions[i] for i in rule]
    return prods_to_eq(prod_seq)