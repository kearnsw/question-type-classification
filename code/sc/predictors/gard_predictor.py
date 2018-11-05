from typing import List

from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader
from allennlp.data.instance import Instance
from allennlp.models import Model
from allennlp.service.predictors.predictor import Predictor


@Predictor.register("sentence-classifier")
class SentenceClassifierPredictor(Predictor):
    """
    Wrapper for the Sentence Classifier class
    """

    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    @overrides
    def _json_to_instance(self, json: JsonDict) -> Instance:
        question = json["Question"]
        label = json["QT"]
        if label not in self._model.vocab._token_to_index["labels"]:
            label = None
        instance = self._dataset_reader.text_to_instance(question, label)
        
        return instance, {}