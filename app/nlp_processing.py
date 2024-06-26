import nltk
from typing import List
from nltk import word_tokenize, pos_tag, ne_chunk, sent_tokenize

nltk.download("punkt")
nltk.download("words")
nltk.download("maxent_ne_chunker")
nltk.download("averaged_perceptron_tagger")


def tokenize_text(text: str) -> List[str]:
    tokens = word_tokenize(text)
    return tokens


def pos_tag_text(text: str) -> List[List[str]]:
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    return pos_tags


def extract_entities(text: str) -> List[List[str]]:
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

    return entities
