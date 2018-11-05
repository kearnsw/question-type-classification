from typing import Dict
import logging

from overrides import overrides

import pandas as pd

from allennlp.common import Params
from allennlp.data.instance import Instance
from allennlp.data.fields import TextField, LabelField
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.dataset_readers import DatasetReader

logger = logging.getLogger(__name__)


@DatasetReader.register("ya")
class YahooDatasetReader(DatasetReader):
    """
    Loads GARD dataset in a method suitable for the ``SentenceClassifier`` model, or any model with a matching API.

    Expected input format is an XML file containing <SubQuestion> tags.

    The output of ``read`` is a list of ``Instance``s with the fields:
        question: ``TextField`` and
        intent: ``TextField``

    Parameters
    ----------
    tokenizer: ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults to WordTokenizer()``.
    token_indexers: ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    """

    def __init__(self, tokenizer: Tokenizer = None, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__()
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        data = pd.read_csv(file_path, delimiter="\t")
        answer_types = ["factoid", "list", "summary", "yn", "yn; list", "yn; factoid", "yn; summary"]
        resource_types = ["Anatomy", "Cause", "Complication", "Diagnosis", "Information", "Management", "Manifestation",
                          "NotDisease", "Other", "OtherEffect", "PersonOrg", "Susceptibility"]
        for qt, at, question in zip(data["QT"], data["AT"], data["Question"]):
            if at in answer_types:
                yield self.text_to_instance(text=question, qt=at)
            """
            if not isinstance(at, float):
                at = at.strip().split(";")
                if len(at) > 1:
                    at = [t for t in at if t not in ["yn"]]         # Remove the superficial yn
                elif at[0] not in answer_types:
                    continue

                if len(at) > 1:                                     # Ensure there aren't two major types
                    continue
                else:
                    yield self.text_to_instance(text=question, qt=at[0])
            """
    @overrides
    def text_to_instance(self, text: str, qt: str) -> Instance:
        tokens = self._tokenizer.tokenize(text)

        question = TextField(tokens, self._token_indexers)
        intent = LabelField(qt, label_namespace="labels")
        return Instance({"question": question, "intent": intent})

    @classmethod
    def from_params(cls, params: Params) -> 'YahooDatasetReader':
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        params.assert_empty(cls.__name__)

        return cls(tokenizer=tokenizer, token_indexers=token_indexers)
