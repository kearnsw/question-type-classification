from typing import Optional, Dict
import overrides

import torch
import numpy as np
from torch.nn.functional import softmax, relu
from torch.nn import Linear
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.nn.initializers import InitializerApplicator
from allennlp.nn.regularizers import RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.modules import FeedForward
from allennlp.modules import Seq2VecEncoder
from allennlp.common import Params
from allennlp.nn.util import get_text_field_mask
from allennlp.common.checks import ConfigurationError


@Model.register("sentence_classifier")
class SentenceClassifier(Model):
    """
    This ``Model`` implements the Sentence Classification model described in `"Convolutional Neural Networks for
    Sentence Classification" <https://www.semanticscholar.org/paper/Convolutional-Neural-Networks-for-Sentence-Classif-Kim/398dee13b3aaaefdf14c78cc1e00dcf265795fd3>`_
    by Yoon Kim, 2014.

    Parameters
    ----------
    vocab: ``Vocabulary``
    text_field_embedder: ``TextFieldEmbedder``
        Used to embed the ``sentence`` ``TextField`` that are sent as input to the model.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 question_encoder: Seq2VecEncoder,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder

        self.nb_classes = self.vocab.get_vocab_size("labels")
        self.question_encoder = question_encoder
        self.enc_dropout = torch.nn.Dropout(0.5)
        self.classifier_feedforward = Linear(question_encoder.get_output_dim(), self.nb_classes)
        self.ff_dropout = torch.nn.Dropout(0.5)
        self.metrics = {
            "accuracy": CategoricalAccuracy(),
        }
        self.loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'SentenceClassifier':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)

        question_encoder = Seq2VecEncoder.from_params(params.pop("question_encoder"))

        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   question_encoder=question_encoder,
                   initializer=initializer,
                   regularizer=regularizer)

    def forward(self,
                question: Dict[str, torch.LongTensor],
                intent: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        q_emb = self.text_field_embedder(question)
        q_mask = get_text_field_mask(question)
        q_enc = self.question_encoder(q_emb, q_mask)
        output_dict = {}
        # output_dict = {"cnn_encoding": q_enc}

        logits = self.classifier_feedforward(q_enc)
        class_probs = softmax(logits)
        output_dict.update({"class_probabilities": class_probs})

        if intent is not None:
            loss = self.loss(logits, intent.squeeze(-1))
            for metric in self.metrics.values():
                metric(logits, intent.squeeze(-1))
            output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        predictions = output_dict['class_probabilities'].cpu().data.numpy()
        argmax_indices = np.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict['label'] = labels
        output_dict.pop('class_probabilities')
        output_dict.pop('loss')
        return output_dict

