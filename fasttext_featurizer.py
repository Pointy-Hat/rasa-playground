import typing
from typing import Any, Optional, Text, Dict, List, Type

import os
import pickle
import numpy as np

from rasa.nlu.components import Component
from rasa.nlu.featurizers.featurizer import DenseFeaturizer
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import Message, TrainingData
from rasa.nlu.featurizers.dense_featurizer.mitie_featurizer import MitieFeaturizer
from rasa.nlu.constants import (
    TEXT,
    DENSE_FEATURE_NAMES,
    TOKENS_NAMES,
)

import fasttext

if typing.TYPE_CHECKING:
    from rasa.nlu.model import Metadata

class FasttextFeaturizer(DenseFeaturizer):
    name="fasttext"
    requires=['text']
    provides=["dense_features"]
    language_list = ["cs"]
    MODEL_LOCATION = "models/model_syn.bin"
    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["sent2vec"]

    @classmethod
    def required_components(cls) -> List[Type["Component"]]:
        return []

    def __init__(self, component_config=None):
        super(FasttextFeaturizer, self).__init__(component_config)
        self.config = {"location":self.MODEL_LOCATION}
        self.model = fasttext.FastText.load_model(self.MODEL_LOCATION)

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Optional[Text] = None,
        model_metadata: Optional["Metadata"] = None,
        cached_component: Optional["Component"] = None,
        **kwargs: Any,
    ) -> "Component":
        return FasttextFeaturizer()


    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        print("I AM ETERNAL!")
        return


    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        #does not really train, uses pre-trained vectors
        for message in training_data.training_examples:
            ftr = self.model.get_sentence_vector(message.get(TEXT))
            ftr=ftr[..., np.newaxis]
            message.set(DENSE_FEATURE_NAMES[TEXT], ftr)

    def process(self, message: Message, **kwargs: Any) -> None:
        ftr = self.model.get_sentence_vector(message.get(TEXT))
        ftr=ftr[..., np.newaxis]
        message.set(DENSE_FEATURE_NAMES[TEXT], ftr)