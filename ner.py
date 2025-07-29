import nltk
from nltk import word_tokenize, pos_tag, ne_chunk, sent_tokenize
from nltk.tree import Tree

input = "Barack Obama went as a prime minister of USA in the year of 2015"
print(len(sent_tokenize(input)))

ner = ne_chunk(pos_tag(word_tokenize(input)))
print(ner)  ## S in the output represented as ROOT

named_entity = []

for subtree in ner:
    if isinstance(subtree, Tree):
        entity = " ".join([token for token, pos in subtree.leaves()])
        named_entity.append(entity)  # ✅ This line must be inside the if block

print(named_entity)


#pip install spacy
import spacy
nlp = spacy.load("en_core_web_sm")

text = "Barack Obama went as a prime minister of USA in the year of 2015 . PM MODI is the prime minister of INDIA."
doc = nlp(text)
named_entity = []
for ent in doc.ents:
    named_entity.append(ent.text)
print(named_entity)