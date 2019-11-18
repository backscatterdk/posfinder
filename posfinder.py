import argparse
import string
from pathlib import Path
from datetime import date, timedelta
import stanfordnlp
import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar


def process(text, nlp, lang, wanted_pos):
    try:
        text = text.translate(str.maketrans('', '', string.punctuation))
        token = nlp(text)
        words = {word for sent in token.sentences for word in sent.words}
        wanted_words = set(filter(lambda word: word.upos in wanted_pos, words))
        wanted_words = ','.join(word.lemma for word in wanted_words if word)
        return wanted_words
    except:
        return ''


def name(i):
    return str(date(2015, 1, 1) + i * timedelta(days=1))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Choose data and whether parrellize.'
    )
    parser.add_argument('data_file', metavar='D', default='data.csv',
                        help='file to process')
    parser.add_argument('column_name', metavar='C', default='for_nlp',
                        help='column in csv to process')
    parser.add_argument('use_dask', metavar='P',
                        help='''choose whether to run serially with pandas or in
parralel with dask''')
    parser.add_argument('lang', metavar='L', default='da',
                        help='langauge')
    parser.add_argument('wanted_pos', metavar='W', default='NOUN',
                        help='Choose wanted POS tags')

    args = parser.parse_args()

    # Deal with stanford models
    model_dir = Path.home() / 'stanfordnlp_resources'
    # model_list is a list of the correct model of max 1 in len
    model_list = list(model_dir.glob(f'{args.lang}_*_models'))
    if not model_list: # download model
        stanfordnlp.download(args.lang)
    nlp = stanfordnlp.Pipeline(
        processors='tokenize,lemma,pos', lang=args.lang)

    # Either use dask og pandas
    if args.use_dask is not None:
        # Read data
        chunks = pd.read_csv(args.data_file, engine='python',
                             error_bad_lines=False, chunksize=20000)

        for i, chunk in enumerate(chunks):
            df = dd.from_pandas(chunk, npartitions=24)

            df['tagged'] = df[args.column_name].apply(
                lambda x: process(text=x, nlp=nlp, lang=args.lang,
                                  wanted_pos=args.wanted_pos)
                if type(x) == str else '',
                meta=(args.column_name, str))

            with ProgressBar():
                df.compute()

            outfolder = Path('.') / 'output'
            Path.mkdir(outfolder, exist_ok=True)
            df.to_csv(outfolder / f'export_{i}-*.csv', name_function=name)
    else:
            # Read data
        data = pd.read_csv(args.data_file, engine='python',
                           error_bad_lines=False)

        df['tagged'] = df[args.column_name].apply(
            lambda x: process(text=x, nlp=nlp, lang=args.lang,
                              wanted_pos=args.wanted_pos)
            if type(x) == str else '')

        outfolder = Path('.') / 'output'
        Path.mkdir(outfolder, exist_ok=True)
        df.to_csv(outfolder / 'export.csv')
