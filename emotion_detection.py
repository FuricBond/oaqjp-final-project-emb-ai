
"""Detect emotions in text using the Watson NLP service."""

import json
import requests

URL = (
    "https://sn-watson-emotion.labs.skills.network/"
    "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
)

HEADERS = {
    "grpc-metadata-mm-model-id":
        "emotion_aggregated-workflow_lang_en_stock"
}


def emotion_detector(text_to_analyze):
    """Return five emotion scores and the dominant emotion."""
    payload = {"raw_document": {"text": text_to_analyze}}

    response = requests.post(
        URL,
        json=payload,
        headers=HEADERS,
        timeout=30
    )

    if response.status_code == 400:
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None,
        }

    response.raise_for_status()
    result = json.loads(response.text)
    emotions = result["emotionPredictions"][0]["emotion"]

    return {
        "anger": emotions["anger"],
        "disgust": emotions["disgust"],
        "fear": emotions["fear"],
        "joy": emotions["joy"],
        "sadness": emotions["sadness"],
        "dominant_emotion": max(emotions, key=emotions.get),
    }
