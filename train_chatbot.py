import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random
import pandas as pd
from IPython.display import display
import itertools
import os
import csv

# Definisikan hyperparameter yang diperbolehkan
layer1_neurons = list(range(60, 129, 10))  # Minimal 16, maksimal 128 neuron di layer 1
layer2_neurons = list(range(30, 65, 10))    # Minimal 8, maksimal 64 neuron di layer 2
layer3_neurons = [30]                  # Softmax layer dengan 30 neuron
epochs_range = list(range(10, 101, 5))    #Minimal 5, maksimal 100 epoch

# Gabungkan semua kemungkinan kombinasi hyperparameter
hyperparameter_combinations = list(itertools.product(layer1_neurons, layer2_neurons, epochs_range))


import nltk
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
from nltk.corpus import stopwords
indonesia_s = stopwords.words('indonesian')
print(len(indonesia_s), "stopwords bahasa indonesia:", indonesia_s)

data_hasil = [
    ['No','layer 1', 'layer 2', 'layer 3', 'Epoch', 'Akurasi', 'Size File'],
]
file_path_csv = 'data_hasil.csv'
file_path = "chatbot_model.h5"


words=[]
classes = []
documents = []
ignore_letters = ['!', '?', ',', '.']
intents_file = open('intents.json').read()
intents = json.loads(intents_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokenize each word
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        #add documents in the corpus
        documents.append((word, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
print(documents)
# lemmaztize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
print (len(documents), "documents")
# classes = intents
print (len(classes), "classes", classes)
# words = all words, vocabulary
print (len(words), "unique lemmatized words", words)

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)
# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)
        
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")

for i, (layer1, layer2, epoch) in enumerate(hyperparameter_combinations, 1):

    # Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
    # equal to number of intents to predict output intent with softmax
    model = Sequential()
    model.add(Dense(layer1, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(layer2, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))

    # Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model 
    hist = model.fit(np.array(train_x), np.array(train_y), epochs=epoch, batch_size=5, verbose=1)
    akurasi = max(hist.history['accuracy'])
# from matplotlib import pyplot as plt
# plt.plot(hist.history['accuracy'])
# #plt.plot(hist.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()
    model.save('chatbot_model.h5', hist)
    if os.path.exists(file_path):
        # Dapatkan ukuran file dalam bytes
        file_size_bytes = os.path.getsize(file_path)

        # Konversi ukuran file ke kilobytes (KB) atau megabytes (MB) jika diinginkan
        file_size_kb = file_size_bytes / 1024
    else:
        print("File tidak ditemukan.")
    hasil = [layer1,layer2,layer3_neurons[0],epoch,akurasi,file_size_kb]
    data_hasil.append(hasil)
    print(
        f"Kombinasi {i}: Layer 1 Neuron = {layer1}, Layer 2 Neuron = {layer2}, Layer 3 Neuron = {layer3_neurons[0]}, epoch = {epoch}, Akurasi = {akurasi}, file size = {file_size_kb}")

with open(file_path_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data_hasil)
print("model created")
