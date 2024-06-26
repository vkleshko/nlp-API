import nltk
from typing import Dict, List
from nltk import word_tokenize, pos_tag
from fastapi import FastAPI, HTTPException

from schemas import TextData

nltk.download("punkt")

app = FastAPI()


@app.post("/tokenize")
async def tokenize(data: TextData) -> Dict[str, List[str]]:
    text = data.text

    if text is None:
        raise HTTPException(status_code=400, detail="No text provided")

    tokens = word_tokenize(text)

    return {"tokens": tokens}
