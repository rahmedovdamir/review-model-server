from typing import List, Tuple
import numpy as np
from transformers import pipeline
from mlserver import MLModel
from mlserver.codecs import decode_args


class ReviewNLIModel(MLModel):
    async def load(self) -> bool:
        self.pipe = pipeline(
            "zero-shot-classification",
            model="cointegrated/rubert-tiny-bilingual-nli",
        )
        self.ready = True
        return self.ready

    @decode_args
    async def predict(
        self, array_inputs: List[str], candidate_labels: List[str]
    ) -> Tuple[List[str], np.ndarray]:
        text = array_inputs[0]
        result = self.pipe(text, candidate_labels=candidate_labels)
        return result["labels"], np.array(result["scores"])
