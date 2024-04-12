import nltk
import gzip
import re
import string
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import accuracy_score
from textblob import TextBlob
#from nltk.corpus import wordnet
from sklearn.metrics import confusion_matrix, classification_report

print("starting")

# Function to read gzipped training files and return a list of sentences.
def read_train_file_gz(file_path):
    sentence_count = 0  # Counter for sentences
    sentences = []  # List to store sentences
    sentence = []  # List to store individual sentence
    # Reading gzipped file
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Check for end of sentence
            if not line:
                if sentence:
                    sentences.append(sentence)
                    sentence = []
                    sentence_count += 1
            else:
                # Extract token and POS tag
                token, pos_tag, _ = line.split()
                sentence.append((token, pos_tag))
    print(f"Total number of sentences: {sentence_count}")
    return sentences



# Function for sentiment analysis using TextBlob
def extract_sentiment(token):
    sentiment = TextBlob(token).sentiment.polarity
    return 'positive' if sentiment > 0 else 'negative' if sentiment < 0 else 'neutral'

# Function to extract features for each token in a sentence
def extract_features(token, index, sentence):
    features = {
        "token": token,
        "is_first": index == 0,  # Is it the first token in the sentence?
        "is_last": index == len(sentence) - 1,  # Is it the last token in the sentence?
        "prefix-1-3": token[:3],  # First 3 characters of the token
        "suffix-1-3": token[-3:],  # Last 3 characters of the token
        # Adding granularity for prefixes and suffixes
        'prefix-1': token[0] if len(token) > 0 else '',
        'prefix-2': token[:2] if len(token) > 1 else '',
        'suffix-1': token[-1] if len(token) > 0 else '',
        'suffix-2': token[-2:] if len(token) > 1 else '',
        
        # Adding 'is_capitals_inside' feature
        'is_capitals_inside': any(char.isupper() for char in token[1:]),
        
        # Adding 'is_numeric' feature
        'is_numeric': token.isdigit(),
        "is_capitalized": token[0].upper() == token[0],
        #"is_all_caps": token.upper() == token,
        "is_all_lower": token.lower() == token,
        "is_punctuation": token in string.punctuation  # Is the token a punctuation mark?
        
    }

    # Adding bi-gram and tri-gram features
    if index > 0:
        features["prev_token"] = sentence[index-1][0]
        features["bigram"] = f"{sentence[index-1][0]}_{token}"
    if index > 1:
        features["prev_prev_token"] = sentence[index-2][0]
        features["trigram"] = f"{sentence[index-2][0]}_{sentence[index-1][0]}_{token}"

    # Adding POS tag context window
    if index > 0:
        features["prev_tag"] = sentence[index-1][1]
    if index < len(sentence) - 1:
        features["next_tag"] = sentence[index+1][1]

    # Adding cardinal and ordinal number features
    features["is_cardinal"] = bool(re.fullmatch(r'\d+(\.\d+)?', token))
    features["is_ordinal"] = bool(re.fullmatch(r'\d+(th|st|nd|rd)', token))

    # Adding sentiment feature
    #sentiment = extract_sentiment(token)
    #features["sentiment"] = sentiment
 
 
    # Vowel-Consonant Ratio
    #num_vowels = len([char for char in token if char in 'aeiouAEIOU'])
    #num_consonants = len(token) - num_vowels
    #features["vowel_consonant_ratio"] = num_vowels / (num_consonants + 1)  # +1 to avoid division by zero

  

# Enhanced contextual features
    #if index > 1:
    #    features["prev_prev_token"] = sentence[index-2][0]
     #   features["fourgram"] = f"{sentence[index-2][0]}_{sentence[index-1][0]}_{sentence[index][0]}"
    #if index < len(sentence) - 2:
    #    features["next_next_token"] = sentence[index+2][0]
    #    features["fourgram"] = f"{sentence[index][0]}_{sentence[index+1][0]}_{sentence[index+2][0]}"
    
    # Enhanced word shape
    #word_shape = re.sub(r'[A-Z]', 'X', re.sub(r'[a-z]', 'x', re.sub(r'[0-9]', 'd', token)))
    #features["word_shape"] = word_shape
    #features["word_shape_length"] = len(word_shape)

     # Word length
    features["word_length"] = len(token)
    
    # Next-Next Tag
    #if index < len(sentence) - 2:
      # features["next_next_tag"] = sentence[index + 2][1]

# Adding surrounding words and POS tags (context window of size 2)
    if index > 1:
        features["prev_prev_token"] = sentence[index-2][0]
        features["prev_prev_tag"] = sentence[index-2][1]
    if index < len(sentence) - 2:
        features["next_next_token"] = sentence[index+2][0]
        features["next_next_tag"] = sentence[index+2][1]

    # Morphological Clues
    features["has_hyphen"] = '-' in token  # Presence of hyphen
    features["has_apostrophe"] = "'" in token  # Presence of apostrophe
    features["has_digit"] = any(char.isdigit() for char in token)  # Presence of digits
    features["non_ascii"] = any(ord(char) >= 128 for char in token)  # Presence of non-ASCII characters
    return features


# Reading and preparing the data
train_data = read_train_file_gz('train.txt.gz')

# Initialize empty lists for training and development feature vectors
X_train, y_train = [], []
X_dev, y_dev = [], []

# Feature extraction for the training set
for sentence in train_data[:-10]:
    ####augmented_sentence = augment_sentence(sentence)  # Augment the sentence
    for index, (token, pos_tag) in enumerate(sentence):
        features = extract_features(token, index, sentence)
        X_train.append(features)
        y_train.append(pos_tag)

# Feature extraction for the development set
for sentence in train_data[-10:]:
    
    for index, (token, pos_tag) in enumerate(sentence):
        features = extract_features(token, index, sentence)
        X_dev.append(features)
        y_dev.append(pos_tag)

# Vectorize feature dictionaries
vectorizer = DictVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_dev = vectorizer.transform(X_dev)

# Train the Logistic Regression classifier
pipeline = make_pipeline(LogisticRegression(max_iter=1000, C=2, penalty='l1', solver='liblinear'))


pipeline.fit(X_train, y_train)

# Evaluate the model on the development set
y_pred = pipeline.predict(X_dev)
dev_accuracy = accuracy_score(y_dev, y_pred)
print(f"Development accuracy = {dev_accuracy:.4f}")

# Function for POS tagging a new sentence
def pos_tag_sentence(sentence):
    tagged_sentence = []
    tokens = sentence.split()
    pseudo_tagged_tokens = [(token, "PSEUDO") for token in tokens]
    X_new = []
    for index, (token, _) in enumerate(pseudo_tagged_tokens):
        features = extract_features(token, index, pseudo_tagged_tokens)
        X_new.append(features)
    X_new = vectorizer.transform(X_new)
    y_new = pipeline.predict(X_new)
    return list(zip(tokens, y_new))


    # Function to POS tag a text file
def pos_tag_text_file(input_file_path, output_file_path, model):
    with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
        lines = infile.readlines()
        pseudo_tagged_sentence = []
        
        for line in lines:
            line = line.strip()
            if not line:
                if pseudo_tagged_sentence:
                    # Prepare the feature vectors for the sentence.
                    X_new = []
                    for index, (token, _) in enumerate(pseudo_tagged_sentence):
                        if index != 0 and token.upper() == token:  # words that are all in caps
                            features = extract_features(token.lower(), index, pseudo_tagged_sentence)
                        else:
                            features = extract_features(token, index, pseudo_tagged_sentence)
                        X_new.append(features)
                    
                    # Transform features and predict
                    X_new = vectorizer.transform(X_new)
                    y_new = model.predict(X_new)
                    
                    # Write the tagged output to the output file
                    for token, tag in zip([token for token, _ in pseudo_tagged_sentence], y_new):
                        outfile.write(f"{token} {tag}\n")  # Using a single blank space instead of \t
                    
                    # Add a newline to indicate end of sentence
                    outfile.write("\n")
                    
                    # Reset the sentence
                    pseudo_tagged_sentence = []
            else:
                pseudo_tagged_sentence.append((line, "PSEUDO"))

input_file_path = 'unlabeled_test_test.txt'  
output_file_path = 'MyLogisticTaggedOutput.txt'  
pos_tag_text_file(input_file_path, output_file_path, pipeline)
# Generate the confusion matrix
conf_matrix = confusion_matrix(y_dev, y_pred)

# Generate a classification report
class_report = classification_report(y_dev, y_pred)

# Print the confusion matrix and classification report
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Example usage
print(pos_tag_sentence("This is a sample sentence for POS tagging."))
