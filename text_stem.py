input = "Barack Obama went as a prime minister of USA in the year of 2015 . PM MODI is the prime minister of INDIA."
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download necessary nltk resources (only need to do once)
nltk.download('punkt')
nltk.download('stopwords')

input_text = "Barack Obama went as a prime minister of USA in the year of 2015. PM MODI is the prime minister of India."

# Tokenize the input text
tokens = word_tokenize(input_text)

# Remove stopwords
stop_words = set(stopwords.words('english'))
tokens_stopwords = [word for word in tokens if word.lower() not in stop_words]

# Initialize stemmer
ps = PorterStemmer()

# Stem the filtered tokens
stemming = []
for word in tokens_stopwords:
    stemming.append(ps.stem(word))

print(stemming)


#Lemmatizer
from nltk import WordNetLemmatizer
lma = []
for word in tokens_stopwords:
    lma.append(WordNetLemmatizer().lemmatize(word))
print(lma)

#POS Tags
from nltk import pos_tag
print(pos_tag(tokens))