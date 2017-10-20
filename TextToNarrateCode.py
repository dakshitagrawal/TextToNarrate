# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 01:39:19 2017

@author: shyam
"""
import pandas as pd
import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Dropout, TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from nltk import FreqDist
import gensim
from sklearn.metrics import f1_score
#%%
df = pd.read_csv('ner_dataset.csv')
df
#%%
txt = ""
txt += df.iloc[0,df.columns.get_loc('Word')]

for i in xrange(1, df.shape[0]-1):
        txt += " "
        txt += df.iloc[i,df.columns.get_loc('Word')]

with open("Story.txt", "w") as text_file:
    text_file.write(txt)
#%%
no_of_datapoints = 1000
data = open("Story.txt", "r").read()
charactersList = list(set(data))
print len(data)
print len(charactersList)

char_to_index = {char:index for index,char in enumerate(charactersList)}
#%%
characterOneHotList = []

for i in xrange(0, no_of_datapoints):
    chars = list(df.iloc[i, df.columns.get_loc('Word')])
    
    char_oneHot = []
    
    for j in xrange(0, len(chars)):
        char_oneHot.append(char_to_index[chars[j]])
    
    characterOneHotList.append(char_oneHot)

print len(characterOneHotList)
print characterOneHotList[0]
#%%
characterOneHotListPadded = pad_sequences(characterOneHotList)

print characterOneHotListPadded.shape
#%%
sentence_character_list = []
sentence_character = [characterOneHotListPadded[0]]

for i in xrange(1, no_of_datapoints):
    if df['Sentence #'].isnull()[i] == True:
        sentence_character.append(characterOneHotListPadded[i])
    else: 
        sentence_character_list.append(sentence_character)
        sentence_character = []
        sentence_character.append(characterOneHotListPadded[i])

sentence_character_list = np.array(sentence_character_list)

print sentence_character_list.shape
#%%
print len(sentence_character_list[1])
sentence_character_list_padded = pad_sequences(sentence_character_list)
print sentence_character_list_padded.shape
print sentence_character_list_padded[0]
#%%
freqdist = FreqDist(df.iloc[:no_of_datapoints, df.columns.get_loc('Word')])

vocab = []
for key in freqdist:
    if key.isalpha():
        vocab.append(key)

vocab.append('UNKNOWN')
print len(vocab)
#%%
sentences = []
x = vocab.index('UNKNOWN')
wordToIndex = [vocab.index(df.iloc[0, df.columns.get_loc('Word')])]
        
for i in xrange(1, no_of_datapoints):
    word = df.iloc[i, df.columns.get_loc('Word')]

    if df['Sentence #'].isnull()[i] == True:
        if word in vocab:
            wordToIndex.append(vocab.index(word))
        else:
            wordToIndex.append(x)
    else: 
        sentences.append(wordToIndex)
        wordToIndex = []
        if word in vocab:
            wordToIndex.append(vocab.index(word))
        else:
            wordToIndex.append(x)

print sentences
#%%
import pickle

with open('sentenceIndex.txt', 'wb') as fp:
    pickle.dump(sentences, fp)
#%%
import pickle

with open ('sentenceIndex.txt', 'rb') as fp:
    sentences = pickle.load(fp)

print sentences
print len(sentences)
#%%
sentencesPadded = pad_sequences(sentences)
print sentencesPadded.shape

#%%
EMBEDDING_DIM = 300

wordPretrainedModel = gensim.models.Word2Vec.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)  

wordEmbeddingPretrainedMatrix = np.zeros((len(vocab), EMBEDDING_DIM))

for i in xrange(0, len(vocab)):
    if word in 
    wordEmbeddingPretrainedMatrix[i] = (model.wv[vocab[i]])

print wordEmbeddingPretrainedMatrix.shape
#%%
EntityOneHot = pd.get_dummies(df.iloc[:no_of_datapoints,df.columns.get_loc('Tag')])

labels = np.empty((EntityOneHot.shape), dtype = int)

for i in xrange(0,EntityOneHot.shape[1]):
    numbers = np.array(EntityOneHot.iloc[:,i])
    labels[:,i] = numbers
              
print len(labels)
print labels

labels_sequence_list = []
labels_sequence = [labels[0]]
        
for i in xrange(1, no_of_datapoints):
    if df['Sentence #'].isnull()[i] == True:
            labels_sequence.append(labels[i])
    else: 
        labels_sequence_list.append(labels_sequence)
        labels_sequence = []
        labels_sequence.append(labels[i])

labels_sequence_list = np.array(labels_sequence_list)
print labels_sequence_list.shape
#%%
labels_sequence_list_padded = pad_sequences(labels_sequence_list)
print labels_sequence_list_padded.shape
#%%
EMBEDDING_DIM = 300
main_input = Input(shape = (sentence_character_list_padded.shape[1], sentence_character_list_padded.shape[2],),
                   dtype = 'float32', name = 'main_input')
print main_input._keras_shape

character_embedding_intermediate = Embedding(input_dim = len(charactersList), output_dim = 30, 
                                input_length = (sentence_character_list_padded.shape[1], sentence_character_list_padded.shape[2])) (main_input)
print character_embedding_intermediate._keras_shape

LSTM_character_forward = TimeDistributed(LSTM(64, return_sequences = False, go_backwards = False))(character_embedding_intermediate)
LSTM_character_backward = TimeDistributed(LSTM(64, return_sequences = False, go_backwards = True))(character_embedding_intermediate)
character_embedding = keras.layers.concatenate([LSTM_character_forward, LSTM_character_backward])
print character_embedding._keras_shape

auxillary_input = Input(shape = (sentencesPadded.shape[1],), dtype = 'float32', name = 'second_input')
word_pretrained_embedding = Embedding(input_dim = len(vocab), output_dim = EMBEDDING_DIM, 
                                      input_length = sentencesPadded.shape[1])(auxillary_input)
#                                      weights = [wordEmbeddingPretrainedMatrix], trainable = True)  
print word_pretrained_embedding._keras_shape  

word_embedding = keras.layers.concatenate([word_pretrained_embedding, character_embedding])
print word_embedding._keras_shape

LSTM_out_forward = LSTM(128, go_backwards = False, return_sequences = True)(word_embedding)
LSTM_out_backward = LSTM(128, go_backwards = True, return_sequences = True)(word_embedding)
LSTM_out = keras.layers.concatenate([LSTM_out_forward, LSTM_out_backward])
print LSTM_out._keras_shape

dense_out_1 = Dense(256, activation = 'relu')(LSTM_out)
print dense_out_1._keras_shape

dropout_1 = Dropout(0.4)(dense_out_1)

main_output = Dense(labels.shape[1], activation = 'softmax', name = 'main_output')(dropout_1)
print main_output._keras_shape

model = Model(inputs = [main_input, auxillary_input], outputs = [main_output])
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#%%
model.summary()
#%%
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import f1_score

class Metrics(keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs={}):
        predict = np.asarray(self.model.predict([self.validation_data[0], self.validation_data[1]]))
        
        argmax = np.argmax(predict, axis = 2)
        
        predictions = np.zeros((predict.shape[0], predict.shape[1], predict.shape[2]), dtype = int)
        for i in xrange(0,argmax.shape[0]):
            for j in xrange(0, argmax.shape[1]):
                predictions[i][j][argmax[i][j]] = 1
                
        predictions = predictions.reshape((-1,predict.shape[2]))

        targ = self.validation_data[2]
        targ = targ.reshape((-1,predict.shape[2]))

        f1s = f1_score(targ, predictions, average = 'micro')
        print "f1_score = ", f1s
        return
#%%
metrics = Metrics()
x = 32
#model.fit([sentence_character_list_padded, sentencesPadded],[labels_sequence_list_padded], epochs = 3, batch_size = 1)
model.fit([sentence_character_list_padded[:x], sentencesPadded[:x]],[labels_sequence_list_padded[:x]], 
          validation_data = [[sentence_character_list_padded[x:], sentencesPadded[x:]],labels_sequence_list_padded[x:]],
          epochs = 3, batch_size = 1, callbacks = [metrics])