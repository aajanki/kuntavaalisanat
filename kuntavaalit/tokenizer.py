import re
from voikko import libvoikko


class VoikkoTokenizer():
    """Tokenizer that lemmatizes Finnish words."""

    def __init__(self):
        self.voikko = libvoikko.Voikko('fi')
        self.token_pattern = re.compile(r"(?u)\b\w[-\w]+\b",)
        self.token_exceptions = {
            'ilmastonmuutto': 'ilmastonmuutos',
            'infran': 'infra',
            'alan': 'ala',
            'hyvinvoipa': 'hyvinvoiva',
        }

    def __call__(self, text):
        """Tokenize and lemmatize text."""
        return [self.lemmatize_token(t) for t in self.token_pattern.findall(text)]

    def lemmatize_token(self, token):
        """Lemmatize one token using libvoikko."""
        analysis = self._disambiguate(self.voikko.analyze(token))
        if analysis:
            candidate = analysis.get('BASEFORM', token).lower()
            return self.token_exceptions.get(candidate, candidate)
        else:
            return token.lower()

    def _disambiguate(self, analyses):
        if len(analyses) > 1:
            # Prefer non-compound words.
            # e.g. "asemassa" will be lemmatized to "asema" not "ase#massa"
            analyses = sorted(analyses, key=self._is_compound_word)
            return analyses[0]
        elif len(analyses) == 1:
            return analyses[0]
        else:
            return None

    def _is_compound_word(self, analysis):
        structure = analysis.get('STRUCTURE', '')
        return structure.count('=') > 1
