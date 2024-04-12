import gzip  
import re  
import string  
from sklearn.feature_extraction import DictVectorizer  
from sklearn.pipeline import make_pipeline  
from sklearn.svm import SVC  
from sklearn.metrics import accuracy_score  
from textblob import TextBlob  
from sklearn.feature_extraction.text import TfidfVectorizer 
import fuzzy
from sklearn.metrics import confusion_matrix, classification_report
soundex = fuzzy.Soundex(4)  # Initialize with Soundex code length of 4
import pickle





print("starting")

#For external testing set
def read_val_file_txt(file_path):
    sentence_count = 0
    sentences = []
    sentence = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if sentence:
                    sentences.append(sentence)
                    sentence = []
                    sentence_count += 1
            else:
                token, pos_tag, _ = line.split()
                sentence.append((token, pos_tag))
    print(f"Total number of sentences: {sentence_count}")
    return sentences

# Read the validation data
val_data = read_val_file_txt('val_labelled.txt')

# Function to read gzipped training files and return a list of sentences.
def read_train_file_gz(file_path):
    sentence_count = 0
    sentences = [] # Initialize an empty list to store all the sentences.
    sentence = []# Initialize an empty list to store the current sentence being read.
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
             # Check if the line is empty, indicating the end of a sentence.
            if not line:

                # If the sentence list is not empty, add it to the list of sentences.
                if sentence:
                    sentences.append(sentence)
                    sentence = []
                    sentence_count += 1
            else:
                token, pos_tag, _ = line.split() #ignore the third column as instructed
                sentence.append((token, pos_tag))
    print(f"Total number of sentences: {sentence_count}")
    return sentences

#function for ayllable feature
def count_syllables(word):
    word = word.lower()
    syllable_count = 0
    
    # Remove trailing 'e' as it's often silent
    if word[-1] == 'e':
        word = word[:-1]
    
    # Count vowel groups
    syllable_count += len(re.findall(r'[aeiouy]+', word))
    
    # Subtract 1 for a vowel group followed by another vowel group, to correct for diphthongs and triphthongs
    syllable_count -= len(re.findall(r'[aeiouy]{1,2}[aeiouy]+', word))
    
    # Ensure that count is at least 1
    syllable_count = max(syllable_count, 1)
    
    return syllable_count
# Function for sentiment analysis.
def extract_sentiment(token):
    # Sentiment analysis
    sentiment = TextBlob(token).sentiment.polarity
    return 'positive' if sentiment > 0 else 'negative' if sentiment < 0 else 'neutral'

#########This part is commented because it is not gonna work with the training set since it's already in normalize language
#########But could be useful if we get to train on actual dataset similar to the test set
##########def normalize_and_sentiment(token):
    # Text normalization rules
    #############normalized_token = re.sub(r'u+', 'you', token, flags=re.IGNORECASE)
    ##########normalized_token = re.sub(r'gonna', 'going to', normalized_token, flags=re.IGNORECASE)


    # Sentiment analysis
   #### sentiment = TextBlob(normalized_token).sentiment.polarity
    #####return normalized_token, 'positive' if sentiment > 0 else 'negative' if sentiment < 0 else 'neutral'



# Reading and preparing the data
train_data = read_train_file_gz('train.txt.gz')



# Function to extract features for each token.
def extract_features(token, index, sentence):
    features = {
        "token": token,
        "is_first": index == 0,  # Is it the first token in the sentence?
        "is_last": index == len(sentence) - 1,  # Is it the last token in the sentence?
        ##"prefix-1-3": token[:3],  # First 3 characters of the token
        #"suffix-1-3": token[-3:],  # Last 3 characters of the token
        # Adding granularity for prefixes and suffixes
        'prefix-1': token[0] if len(token) > 0 else '',
        'prefix-2': token[:2] if len(token) > 1 else '',
        'prefix-3': token[:3] if len(token) > 2 else '',
        'suffix-1': token[-1] if len(token) > 0 else '',
        'suffix-2': token[-2:] if len(token) > 1 else '',
        'suffix-3': token[-3:] if len(token) > 2 else '',
        
        # Adding 'is_capitals_inside' feature
        'is_capitals_inside': any(char.isupper() for char in token[1:]),
        
        # Adding 'is_numeric' feature
        'is_numeric': token.isdigit(),
        "is_capitalized": token[0].upper() == token[0],
        "is_all_caps": token.upper() == token,
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
    #features["is_cardinal"] = bool(re.fullmatch(r'\d+(\.\d+)?', token))
    features["is_ordinal"] = bool(re.fullmatch(r'\d+(th|st|nd|rd)', token))

    # Adding sentiment feature
    sentiment = extract_sentiment(token)
    features["sentiment"] = sentiment
 
 # Add context words (previous and next) to the features
    features['prev_word'] = '' if index == 0 else sentence[index - 1]
    features['prev_word2'] = '' if index <= 1 else sentence[index - 2]
    features['prev_word3'] = '' if index <= 2 else sentence[index - 3]
    features['next_word'] = '' if index == len(sentence) - 1 else sentence[index + 1]
    features['next_word2'] = '' if index >= len(sentence) - 2 else sentence[index + 2]
    features['next_word3'] = '' if index >= len(sentence) - 3 else sentence[index + 3]
    # Vowel-Consonant Ratio
    #num_vowels = len([char for char in token if char in 'aeiouAEIOU'])
    #num_consonants = len(token) - num_vowels
    #features["vowel_consonant_ratio"] = num_vowels / (num_consonants + 1)  # +1 to avoid division by zero

# TF-IDF Vector
   #tf_idf_vector = tfidf_vectorizer.transform([token]).toarray()[0]
   #features['tf_idf'] = tf_idf_vector.tolist()  # Convert numpy array to list

# Adding Character-level features
    features['char_ngram'] = [token[i:i+3] for i in range(len(token) - 2)]


   
   
    
    # Enhanced word shape
    word_shape = re.sub(r'[A-Z]', 'X', re.sub(r'[a-z]', 'x', re.sub(r'[0-9]', 'd', re.sub(r'[\W_]', 'p', token))))
    features["word_shape"] = word_shape
    #features["word_shape_length"] = len(word_shape)

   

     # Word length
    #features["word_length"] = len(token)
    
    # Next-Next Tag
    if index < len(sentence) - 2:
       features["next_next_tag"] = sentence[index + 2][1]

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
    features["sentence_position_ratio"] = index / len(sentence) #sentence position ratio

    
    features["syllable_count"] = count_syllables(token)#count syllabus

    features["has_special_char"] = any(char in '&%$#@!' for char in token)#special character

    features["has_repeated_letters"] = any(token.count(char) > 1 for char in set(token))#repeat letter

    #phoenic feature
    #features["soundex"] = soundex(token)



    # Syntactic Patterns for 'EX' and 'RP'
    if index > 0 and index < len(sentence) - 1:
        features["prev_next_tags"] = f"{sentence[index-1][1]}_{sentence[index+1][1]}"

    # Lexical Resources for 'FW'
    common_foreign_words = set([
        "bonjour", "amigo", "ciao", "danke", "gracias", "etc", "Etc",
        "si", "oui", "nein", "ja", "niet", "sayonara", "adios", "au revoir",
        "tschüss", "hola", "hallo", "salaam", "shalom", "namaste",
        "konbanwa", "nihao", "merhaba", "privet", "aloha", "ola","de","la","glasnost"
    ]) 
    features["is_common_foreign_word"] = token.lower() in common_foreign_words

    return features


#Initialize empty list for training and development feature vec
X_train, y_train = [], []
X_dev, y_dev = [], []


# Feature extraction for the training set
for sentence in train_data[:-3000]:
    for index, (token, pos_tag) in enumerate(sentence):
        features = extract_features(token, index, sentence)
        X_train.append(features)
        y_train.append(pos_tag)

# Feature extraction for the development set
for sentence in train_data[-3000:]:
    for index, (token, pos_tag) in enumerate(sentence):
        features = extract_features(token, index, sentence)
        X_dev.append(features)
        y_dev.append(pos_tag)

# Vectorize feature dictionaries
vectorizer = DictVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_dev = vectorizer.transform(X_dev)

# Train the SVM classifier
pipeline = make_pipeline(SVC(kernel='rbf', C=1))
pipeline.fit(X_train, y_train)

# Evaluate the model on the development set
y_pred = pipeline.predict(X_dev)
dev_accuracy = accuracy_score(y_dev, y_pred)
print(f"Development accuracy = {dev_accuracy:.4f}")


####for external testing set:
# Initialize list for validation features and labels
X_val, y_val = [], []

# Feature extraction for the validation set
for sentence in val_data:
    for index, (token, pos_tag) in enumerate(sentence):
        features = extract_features(token, index, sentence)
        X_val.append(features)
        y_val.append(pos_tag)

# Vectorize feature dictionaries for validation set
X_val = vectorizer.transform(X_val)

# Function for POS tagging a new sentence
def pos_tag_sentence(sentence):
    tagged_sentence = []
    tokens = sentence.split()
    pseudo_tagged_tokens = [(token, "PSEUDO") for token in tokens]
    
    # Prepare the feature vectors for the new sentence.
    X_new = []
    for index, (token, _) in enumerate(pseudo_tagged_tokens):
        features = extract_features(token, index, pseudo_tagged_tokens)
        X_new.append(features)
     
    #transform feature and predict
    X_new = vectorizer.transform(X_new)
    y_new = pipeline.predict(X_new)
    
    return list(zip(tokens, y_new))


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
                        features = extract_features(token, index, pseudo_tagged_sentence)
                        X_new.append(features)
                    
                    # Transform features and predict
                    X_new = vectorizer.transform(X_new)
                    y_new = model.predict(X_new)
                    
                    # Write the tagged output to the output file
                    for token, tag in zip([token for token, _ in pseudo_tagged_sentence], y_new):
                        outfile.write(f"{token}\t{tag}\n")
                    
                    # Add a newline to indicate end of sentence
                    outfile.write("\n")
                    
                    # Reset the sentence
                    pseudo_tagged_sentence = []
            else:
                # Treat each line as a single token
                pseudo_tagged_sentence.append((line, "PSEUDO"))

# Example usage
input_file_path = 'unlabeled_test_test.txt'  
output_file_path = 'TaggedOutput.txt'
model = pipeline  # Assuming 'pipeline' is your trained model
pos_tag_text_file(input_file_path, output_file_path, model)


# Generate the confusion matrix
conf_matrix = confusion_matrix(y_dev, y_pred)

# Generate a classification report
class_report = classification_report(y_dev, y_pred)

# Print the confusion matrix and classification report
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Evaluate the model on the validation set
y_val_pred = pipeline.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation accuracy = {val_accuracy:.4f}")
# Save the trained model
with open("trained_svm_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

# Save the vectorizer
with open("trained_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)


# Read the raw text file
with open('TaggedOutput.txt', 'r') as f:
    lines = f.readlines()

# Open a new file to write the output
with open('CARTO.test.txt', 'w') as f:
    for line in lines:
        # Strip leading and trailing whitespaces
        stripped_line = line.strip()
        
        # Handle empty lines
        if not stripped_line:
            f.write("\n")
            continue
        
        # Split each line by tab to separate the word and the POS tag
        parts = stripped_line.split('\t')
        
        # Check if the line contains both word and POS tag
        if len(parts) == 2:
            token, pos_tag = parts
            
            # Write the word and POS tag separated by a single space
            f.write(f"{token} {pos_tag}\n")
        else:
            print(f"Skipping malformed line: {stripped_line}")

# Example usage
print(pos_tag_sentence("This is a sample sentence for POS tagging."))
