import nltk
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')

from nltk import word_tokenize, pos_tag, ne_chunk, sent_tokenize
from nltk.tree import Tree

input_text = "Barack Obama went as a prime minister of USA in the year of 2015 . PM MODI is the prime minister of INDIA."
print(len(sent_tokenize(input_text)))

ner = ne_chunk(pos_tag(word_tokenize(input_text)))
print(ner)

named_entity = []
for subtree in ner:
    if isinstance(subtree, Tree):
        entity = " ".join([token for token, pos in subtree.leaves()])
        named_entity.append(entity)  # indentation fixed here

print(named_entity)
