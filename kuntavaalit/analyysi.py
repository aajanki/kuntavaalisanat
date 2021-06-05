import eli5
import itertools
import json
import numpy as np
import pandas as pd
import re
from zipfile import ZipFile
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC
from .tokenizer import VoikkoTokenizer

url_re = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')


def main():
    promises = load_promises()
    parties = load_parties()
    muncipalities = load_muncipalities()
    
    print('Ehdokkaiden lukumäärät')
    candidate_counts = (
        promises[['id', 'party_id']]
        .drop_duplicates()['party_id']
        .value_counts()
        .rename('candidate_counts'))
    candidate_counts.index = candidate_counts.index.map(parties)
    print(candidate_counts)
    print()
    
    print('Osuus ehdokkaista, joilla vähintään yksi vaalilupaus')
    has_any_promise_by_party = proportion_of_candidates_with_promise(promises)
    has_fi_promise_by_party = (proportion_of_candidates_with_promise(
        promises[['id', 'party_id', 'constituency_id', 'promise_fi']])
                               .rename(columns={'has_promise': 'has_promise_finnish'}))
    has_promise_by_party = pd.concat([has_any_promise_by_party, has_fi_promise_by_party], axis=1)
    has_promise_by_party.index = has_promise_by_party.index.map(parties).rename('party')
    print(has_promise_by_party)
    print()

    print('Puolueille ominaiset sanat')
    characteristic = find_characteristic_words(promises, parties, muncipalities)
    for party_name, weighted_words in characteristic.items():
        print(party_name)
        print('  ' + '\n  '.join(f'{word} ({weight:.2})' for word, weight in weighted_words))
        print()


def find_characteristic_words(promises, parties, muncipalities, k=10):
    # Only Finnish non-duplicated promises
    df = (promises[['id', 'party_id', 'promise_fi']]
          .dropna(subset=['promise_fi'])
          .drop_duplicates()[['party_id', 'promise_fi']])

    tokenizer = VoikkoTokenizer()
    stop_words = [tokenizer(w)[0] for w in get_stop_words(muncipalities)]
    vec = CountVectorizer(min_df=20, max_df=0.1,
                          preprocessor=text_cleanup,
                          tokenizer=tokenizer,
                          stop_words=stop_words)
    X = vec.fit_transform(df['promise_fi'])

    party_bows = {}
    for pid in df['party_id'].unique():
        # Mean bag-of-word vectors for party's candidates. Normalizes
        # the number of candidates per party.
        #
        # How likely a word is to occur in an average promise?
        mean_bow = X[df['party_id'] == pid, :].mean(axis=0)
        party_bows[parties[pid]] = np.squeeze(np.asarray(mean_bow))
    bows = pd.DataFrame(party_bows, index=vec.get_feature_names())

    # Normalizing by the promise text length?
    
    # The proportion of word usages in party's (average) promise
    # versus all parties
    props = bows.divide(bows.sum(axis=1), axis=0)

    # List the k most characteristic words and their weights
    words = {}
    for party_name, p in props.iteritems():
        words[party_name] = (p.rename('weight')
                             .rename_axis('word')
                             .sort_values(ascending=False)[:k]
                             .reset_index()
                             .to_records(index=False).tolist())
        
    return words


def find_characteristic_words_svm(promises, parties):
    # Only Finnish non-duplicates
    df = (promises[['id', 'party_id', 'promise_fi']]
          .dropna(subset=['promise_fi'])
          .drop_duplicates()[['party_id', 'promise_fi']])

    # Combine small parties and sitoutumattomat into an "other"
    # category with ID 99
    promise_counts = df['party_id'].value_counts()
    small_parties = (promise_counts[promise_counts < 1000].index.to_list() +
                     parties[parties == 'Sitoutumaton'].index.to_list())
    df['party_id'] = df['party_id'].replace(small_parties, 99)

    tokenizer = VoikkoTokenizer()
    vec = TfidfVectorizer(min_df=10, max_df=0.1, tokenizer=tokenizer, use_idf=True)
    clf = LinearSVC(C=0.001, loss='hinge', intercept_scaling=4.0,
                    max_iter=100000, multi_class='ovr')
    scaler = MaxAbsScaler()
    pipe = make_pipeline(vec, scaler, clf)
    pipe.fit(df['promise_fi'], df['party_id'])

    print(classification_report(df['party_id'], pipe.predict(df['promise_fi'])))

    target_names = [parties[x] if x != 99 else 'Muut' for x in clf.classes_]
    html = eli5.show_weights(clf, vec=vec, top=20, target_names=target_names)
    with open('results.html', 'w') as outf:
        outf.write(html.data)


def proportion_of_candidates_with_promise(promises):
    return (promises
        .drop(columns=['constituency_id'])
        .groupby(['id', 'party_id'])
        .any()
        .any(axis=1)
        .rename('has_promise')
        .reset_index()
        .drop(columns=['id'])
        .groupby('party_id')
        .mean())

        
def get_stop_words(muncipalities):
    # mucipality names and -lAinen nouns
    muncipalities_lo = [x.lower() for x in muncipalities if ' ' not in x]
    stop_words = list(itertools.chain.from_iterable(
        [name, name + 'lainen', name + 'läinen']
        for name in muncipalities_lo)) + ['lahtelainen']
    return stop_words


def detect_language(text):
    # A really simple heuristics.
    #
    # Fails for example if the input contains more than one langauge.
    if not text:
        return None

    se = re.compile(r'\b(?:på|för|och|som|skall|vill|stad|inom|vår|det)\b', re.IGNORECASE)
    en = re.compile(r'\b(?:promise|want)\b', re.IGNORECASE)

    if se.search(text):
        return 'se'
    elif en.search(text):
        return 'en'
    else:
        return 'fi'


def text_cleanup(text):
    # Drop URLs
    return url_re.sub(' ', text)


def load_promises():
    promises = []
    with ZipFile('data/vaalikone/items.zip') as zf:
        for zipinfo in zf.infolist():
            basename = zipinfo.filename.split('/')[-1]
            if basename.startswith('candidates_'):
                with zf.open(zipinfo) as f:
                    candidate = json.load(f)
                    
                    for i in range(1, 4):
                        data = {
                            'id': candidate['id'],
                            'party_id': candidate['party_id'],
                            'constituency_id': candidate['constituency_id']
                        }
                        promise = candidate.get('info', {}).get(f'election_promise_{i}')
                        if promise:
                            data.update(promise)

                            # Try to fix incorrectly labelled languages
                            lang = detect_language(data['fi'])
                            if lang and lang != 'fi':
                                if not data[lang]:
                                    data[lang] = data['fi']
                                del data['fi']
                            
                        promises.append(data)

    column_name_map = {
        'fi': 'promise_fi',
        'se': 'promise_se',
        'en': 'promise_en',
        'ru': 'promise_ru',
        'sme': 'promise_sme',
    }
    return pd.DataFrame(promises).rename(columns=column_name_map)


def load_parties():
    parties = {}
    with ZipFile('data/vaalikone/items.zip') as zf:
        for zipinfo in zf.infolist():
            basename = zipinfo.filename.split('/')[-1]
            if basename == 'parties.json':
                with zf.open(zipinfo) as f:
                    for data in json.load(f):
                        party_id = data['id']
                        name = data['name_fi']
                        existing_name = parties.get('party_id')
                        if existing_name:
                            assert name == existing_name
                        else:
                            parties[party_id] = name

    return pd.Series(parties).rename('party_name')


def load_muncipalities():
    with open('data/kunnat/kunnat.txt') as f:
        return [x.strip() for x in f.readlines()]


if __name__ == '__main__':
    main()
