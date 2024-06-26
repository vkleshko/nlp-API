import nltk
from typing import Dict, List
from fastapi import FastAPI, HTTPException
from nltk import word_tokenize, pos_tag, ne_chunk, sent_tokenize

from schemas import TextData

nltk.download("punkt")
nltk.download("words")
nltk.download("maxent_ne_chunker")
nltk.download("averaged_perceptron_tagger")

app = FastAPI()


@app.post("/tokenize")
async def tokenize(data: TextData) -> Dict[str, List[str]]:
    text = data.text

    if text is None:
        raise HTTPException(status_code=400, detail="No text provided")

    tokens = word_tokenize(text)

    return {"tokens": tokens}


@app.post("/pos_tag")
async def pos_tagging(data: TextData) -> Dict[str, List[List[str]]]:
    text = data.text

    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)

    return {"pos_tags": pos_tags}


@app.post("/ner")
async def named_entity_recognition(data: TextData) -> Dict[str, List[List[str]]]:
    text = data.text

    if text is None:
        raise HTTPException(status_code=400, detail="No text provided")

    entities = []
    for sent in sent_tokenize(text):
        tokens = nltk.word_tokenize(sent)
        pos_tags = nltk.pos_tag(tokens)
        named_entities = ne_chunk(pos_tags)

        for chunk in named_entities:
            if hasattr(chunk, "label"):
                entity_type = chunk.label()
                entity = "".join(c[0] for c in chunk)

                entities.append([entity, entity_type])

    return {"entities": entities}
