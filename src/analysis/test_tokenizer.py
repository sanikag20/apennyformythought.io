
import spacy

nlp = spacy.load("en_core_web_sm")

examples = [
    "I'm going home, but I don't know why.",
    "It's Wednesday, my dudes!",
    "The world's smallest violin isn't playing.",
    "I've read it, and I'm still thinking about it.",
]

for text in examples:
    doc = nlp(text)

    print(f"\nOriginal: {text}")
    print("Tokens:", [token.text for token in doc])
    print(
        "Lemmas:",
        [(token.text, token.lemma_) for token in doc]
    )
