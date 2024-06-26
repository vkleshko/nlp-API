from fastapi import FastAPI, HTTPException
from typing import Dict, List
from app.schemas import TextData

from app.nlp_processing import tokenize_text, pos_tag_text, extract_entities

app = FastAPI()


@app.post("/tokenize")
async def tokenize(data: TextData) -> Dict[str, List[str]]:
    text = data.text
    if text is None:
        raise HTTPException(status_code=400, detail="No text provided")
    tokens = tokenize_text(text)
    return {"tokens": tokens}


@app.post("/pos_tag")
async def pos_tagging(data: TextData) -> Dict[str, List[List[str]]]:
    text = data.text
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    pos_tags = pos_tag_text(text)
    return {"pos_tags": pos_tags}


@app.post("/ner")
async def named_entity_recognition(data: TextData) -> Dict[str, List[List[str]]]:
    text = data.text
    if text is None:
        raise HTTPException(status_code=400, detail="No text provided")
    entities = extract_entities(text)
    return {"entities": entities}
