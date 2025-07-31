from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Sample text
text = "Barack Obama went as a prime minister of USA in the year of 2015."

# Tokenize
tokens = word_tokenize(text)

# Define stopwords set
stop_words = set(stopwords.words('english'))

# Remove stopwords
tokens_stopwords = [word for word in tokens if word.lower() not in stop_words]

# Initialize stemmer once
stemmer = PorterStemmer()

# Stem the filtered tokens
stemming = [stemmer.stem(word) for word in tokens_stopwords]

print(stemming)


#Lemmatizer
from nltk import WordNetLemmatizer
lma = []
for word in tokens_stopwords:
    lma.append(WordNetLemmatizer().lemmatize(word))
print(lma)

#POS Tags
from nltk import pos_tag
from nltk.tokenize import word_tokenize

text = "Barack Obama went to the USA in 2015."  # Define some sample text
word_tokens = word_tokenize(text)  # This line is required
print(pos_tag(word_tokens))  # This will print the POS tags
