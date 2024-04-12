import gzip
import nltk
import re
import string
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from collections import Counter

confusion_matrix = Counter()

# Reading the training data
def read_train_file_gz(file_path):
    sentence_count = 0
    sentences = [] # List to hold the sentences
    sentence = [] # Temp list to hold tokens and POS tags for each sentence
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if sentence:
                    sentences.append(sentence)
                    sentence = [] #reset temp list
                    sentence_count += 1
            else:
                token, pos_tag, _ = line.split()
                sentence.append((token, pos_tag))
    print(f"Total number of sentences: {sentence_count}")
    return sentences

# Feature extraction for training and tagging
def extract_features(token, index, sentence):
    features = {
        "token": token,
        "is_first": index == 0,
        "is_last": index == len(sentence) - 1,
        #pre and suffix feature
        "prefix-1-3": token[:3],  
        "suffix-1-3": token[-3:],  
        #More Lexical Features: Introduce features like whether the word is uppercase, whether it contains a digit, or if it's a common punctuation
        "is_punctuation": token in string.punctuation,  # Check for punctuation
        #"is_upper": token.isupper(),
        #"is_lower": token.islower(),
        #"is_mixed_case": not token.isupper() and not token.islower()
    }
    
    # Adding bi-gram feature
    if index > 0:
        features["prev_token"] = sentence[index-1][0]
        features["bigram"] = f"{sentence[index-1][0]}_{token}"
        
    # Adding tri-gram feature
    if index > 1:
        features["prev_prev_token"] = sentence[index-2][0]
        features["trigram"] = f"{sentence[index-2][0]}_{sentence[index-1][0]}_{token}"

    # Tag Context: Incorporate the POS tags of previous and next words as additional features. Since your data follows a sequence, this context could be useful.
    if index > 0:
        features["prev_tag"] = sentence[index-1][1]
    if index < len(sentence) - 1:
        features["next_tag"] = sentence[index+1][1]
    
    # Detect cardinal numbers by checking if the token is purely numerical
    features["is_cardinal"] = bool(re.fullmatch(r'\d+(\.\d+)?', token))
    
    # Detect ordinal numbers by looking for common suffixes like "th", "st", "nd", "rd"
    features["is_ordinal"] = bool(re.fullmatch(r'\d+(th|st|nd|rd)', token))

    return features

# Reading the data
train_data = read_train_file_gz('train.txt.gz')

train_set = []
dev_set = []

# Preparing training and set
for sentence in train_data[:-1380]:
    for index, (token, pos_tag) in enumerate(sentence): #enumerate for index
        #extract and append feature for each token
        features = extract_features(token, index, sentence)
        train_set.append((features, pos_tag))

#Preparing development set
for sentence in train_data[-1380:]:
    for index, (token, pos_tag) in enumerate(sentence): #enumerate for index
        #extract and append feature for each token
        features = extract_features(token, index, sentence)
        dev_set.append((features, pos_tag))

# Model training
classifier = NaiveBayesClassifier.train(train_set)

correct_tags = Counter()
total_tags = Counter()

# Model evaluation
# Model evaluation
correct_count = 0
total_count = 0

for features, actual_tag in dev_set:
    predicted_tag = classifier.classify(features)
    
    if actual_tag == predicted_tag:
        correct_count += 1
        correct_tags[actual_tag] += 1  # Increment counter for the correct tag
        
    total_count += 1
    total_tags[actual_tag] += 1  # Increment counter for the total occurrences of the tag

print(f"Overall Accuracy: {correct_count / total_count}")

# Calculate and display accuracy for each POS tag
for tag, total in total_tags.items():
    correct = correct_tags[tag]
    accuracy = correct / total
    print(f"Accuracy for tag {tag}: {accuracy}")


# Printing the confusion matrix
for (actual_tag, predicted_tag), count in confusion_matrix.items():
    print(f"Actual: {actual_tag}, Predicted: {predicted_tag}, Count: {count}")


# POS tagging function for new sentence
def pos_tag_sentence(sentence):
    tagged_sentence = []  
    tokens = sentence.split()  # Split sentence into tokens
    pseudo_tagged_tokens = [(token, "PSEUDO") for token in tokens]  # Assign a pseudo POS tag to each token
    
    # Loop through each pseudo-tagged token in the sentence
    for index, (token, _) in enumerate(pseudo_tagged_tokens):
        #used extract feature func and the trained classifier to get the pos tag for each token  
        features = extract_features(token, index, pseudo_tagged_tokens)  
        pos_tag = classifier.classify(features)  
        tagged_sentence.append((token, pos_tag))  #append to the tagged sentence
    return tagged_sentence


# example usage
print(pos_tag_sentence("This is a sample sentence for POS tagging."))




