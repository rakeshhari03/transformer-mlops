import os
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

class ModelService:

    def __init__(self, model_path):

        self.device = (
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(model_path)
        )

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(model_path)
        )

        self.model.to(self.device)

        self.model.eval()

    def predict(self, text):

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():

            outputs = self.model(**inputs)

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1
            )

        predicted_class = probabilities.argmax(
            dim=-1
        ).item()

        confidence = probabilities[
            0, predicted_class
        ].item()

        return {
            "prediction": predicted_class,
            "confidence": confidence
        }