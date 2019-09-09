"""
This script takes in a csv with a column of text and generates
a the same csv with words of a given PoS-tag (eg nouns, verbs etc)
filtered in the column "filtered"

For linux systems, you may have to run this in your terminal first
to get the picking options to work

$ export TERM=linux
$ export TERMINFO=/bin/zsh

"""

import os
import string
import pandas as pd
import stanfordnlp
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
from pick import pick


def clean_text(text):
    """
    Remove punctuation
    """
    return text.translate(str.maketrans('', '', string.punctuation))


def get_lemma(token):
    """
    Get the lemma from the tokeniszed sentences
    """
    return [word.lemma for sent in token.sentences
            for word in sent.words]


def remove_stop(words):
    """
    Remove stop words
    """
    return [word for word in words if word not in STOP_WORDS]


def filter_pos(token):
    """
    This is for filtering based on word type
    """
    filtered = []
    for sent in token.sentences:
        filtered.extend([word.lemma for word in sent.words
                         if word.upos in WANTED_POS])
    filtered = list(set(filtered))
    return filtered


def remove_punc(words):
    """
    Removes punctuation and lowercases.
    """
    out = []
    for w in words:
        out.append(''.join(e.lower() for e in w if e.isalnum()))
    return out


nltk.download('stopwords')

"""
Choose which types of words (eg. nouns, verbs) are desired.
For POS tags, see https://universaldependencies.org/u/pos/
"""

# Pick which PoS tags you want
POSTAG_TITLE = 'Please POS tags (SPACE to mark, ENTER to continue)'
POSTAGS = ['ADJ', 'ADP', 'PUNCT', 'ADV', 'AUX', 'SYM', 'INTJ', 'CCONJ',
           'X', 'NOUN', 'DET', 'PROPN', 'NUM', 'VERB', 'PART', 'PRON', 'SCONJ']
WANTED_POS = pick(POSTAGS, POSTAG_TITLE,
                  multi_select=True, min_selection_count=1)

WANTED_POS = [pos[0] for pos in WANTED_POS]

# Pick language
LANG_TITLE = 'Please chose which language the text is in.'
LANGS = ['en', 'da', 'other']
LANG, LANG_TITLE = pick(LANGS, LANG_TITLE)
if LANG == 'other':
    LANG = input('Please input language code \
(see stanfordnlp.github.io/stanfordnlp/models.html)')

# Download model for nlp.
if not os.path.exists(os.path.join(os.environ['HOME'],
                                   'stanfordnlp_resources', f'{LANG}_ddt_models')):
    stanfordnlp.download(F'{LANG}')


# Set up nlp pipeline
NLP = stanfordnlp.Pipeline(processors='tokenize,mwt,lemma,pos', lang='da')

# Read data. Change to correspond.
DATA = pd.read_csv('parents_full.csv')

# For progress bar
tqdm.pandas()

# Get get stop words -> doesn't seem to be used
STOP_WORDS = stopwords.words('danish')

# Pick column for terms
COLUMN_TITLE = 'Please chose which column contains the words.'
COLUMNS = DATA.columns
COLUMN, COLUMN_TITLE = pick(COLUMNS, COLUMN_TITLE)

DATA['tokens'] = DATA[COLUMN].progress_apply(lambda text: NLP(text))
DATA['lemmas'] = DATA['tokens'].apply(get_lemma)
DATA['lemmas_string'] = DATA['lemmas'].apply(lambda x: " ".join(x))
DATA['without_stop'] = DATA['lemmas'].apply(remove_stop)
DATA['filtered'] = DATA['tokens'].apply(filter_pos)
DATA['filtered'] = DATA['filtered'].apply(remove_punc)
DATA['filtered'] = DATA['filtered'].apply(lambda x: ", ".join(x))
DATA.drop(['tokens', 'lemmas', 'lemmas_string', 'without_stop'],
          axis=1, inplace=True)

DATA.to_csv('pos_tagged.csv')
