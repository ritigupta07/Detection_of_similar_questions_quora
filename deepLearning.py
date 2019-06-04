import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import binarize

from keras.layers import *
from keras.models import Model
from keras.layers import merge as Merge
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU 
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import TimeDistributed, Lambda
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers.advanced_activations import PReLU
from keras.preprocessing import sequence, text

data = pd.read_csv('data/quora_duplicate_questions.tsv', sep='\t')

y = data.is_duplicate.values

t = text.Tokenizer(nb_words=200000)

t.fit_on_texts(list(data.question1.values.astype(str)) + list(data.question2.values.astype(str)))

maxLen = 40

x1 = t.texts_to_sequences(data.question1.values.astype(str))
x1 = sequence.pad_sequences(x1, maxlen=maxLen)

x2 = t.texts_to_sequences(data.question2.values.astype(str))
x2 = sequence.pad_sequences(x2, maxlen=maxLen)

X = np.hstack((x1,x2))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

wordIdx = t.word_index

embeddingsIdx = {}
f = open('data/glove.840B.300d.txt')
for line in tqdm(f):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddingsIdx[word] = coefs
f.close()

print('Embedding Index of %s word vectors created.' % len(embeddingsIdx))

embeddingMatrix = np.zeros((len(wordIdx) + 1, 300))
for word, i in tqdm(wordIdx.items()):
    embeddingVec = embeddingsIdx.get(word)
    if embeddingVec is not None:
        embeddingMatrix[i] = embeddingVec

print('Embedding Matrix Created.')

model = Sequential()
model.add(Embedding(len(wordIdx) + 1,
                     300,
                     weights=[embeddingMatrix],
                     input_length=40,
                     trainable=False))

model.add(TimeDistributed(Dense(300, activation='relu')))
model.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(300,)))

model2 = Sequential()
model2.add(Embedding(len(wordIdx) + 1,
                     300,
                     weights=[embeddingMatrix],
                     input_length=40,
                     trainable=False))

model2.add(TimeDistributed(Dense(300, activation='relu')))
model2.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(300,)))

model3 = Sequential()
model3.add(Embedding(len(wordIdx) + 1,
                     300,
                     weights=[embeddingMatrix],
                     input_length=40,
                     trainable=False))

filter_length = 5
nb_filter = 64

model3.add(Convolution1D(nb_filter=nb_filter,
                         filter_length=filter_length,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1))
model3.add(Dropout(0.2))

model3.add(Convolution1D(nb_filter=nb_filter,
                         filter_length=filter_length,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1))

model3.add(GlobalMaxPooling1D())
model3.add(Dropout(0.2))

model3.add(Dense(300))
model3.add(Dropout(0.2))
model3.add(BatchNormalization())

model4 = Sequential()
model4.add(Embedding(len(wordIdx) + 1,
                     300,
                     weights=[embeddingMatrix],
                     input_length=40,
                     trainable=False))
model4.add(Convolution1D(nb_filter=nb_filter,
                         filter_length=filter_length,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1))
model4.add(Dropout(0.2))

model4.add(Convolution1D(nb_filter=nb_filter,
                         filter_length=filter_length,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1))

model4.add(GlobalMaxPooling1D())
model4.add(Dropout(0.2))

model4.add(Dense(300))
model4.add(Dropout(0.2))
model4.add(BatchNormalization())
model5 = Sequential()
model5.add(Embedding(len(wordIdx) + 1, 300, input_length=40, dropout=0.2))
model5.add(LSTM(300, dropout_W=0.2, dropout_U=0.2))

model6 = Sequential()
model6.add(Embedding(len(wordIdx) + 1, 300, input_length=40, dropout=0.2))
model6.add(LSTM(300, dropout_W=0.2, dropout_U=0.2))

MergedModel1 = add([model.output, model2.output, model3.output, model4.output, model5.output, model6.output])
MergedModel = Sequential()
MergedModel.add(BatchNormalization())

MergedModel.add(Dense(300))
MergedModel.add(PReLU())
MergedModel.add(Dropout(0.2))
MergedModel.add(BatchNormalization())

MergedModel.add(Dense(300))
MergedModel.add(PReLU())
MergedModel.add(Dropout(0.2))
MergedModel.add(BatchNormalization())

MergedModel.add(Dense(300))
MergedModel.add(PReLU())
MergedModel.add(Dropout(0.2))
MergedModel.add(BatchNormalization())

MergedModel.add(Dense(300))
MergedModel.add(PReLU())
MergedModel.add(Dropout(0.2))
MergedModel.add(BatchNormalization())

MergedModel.add(Dense(300))
MergedModel.add(PReLU())
MergedModel.add(Dropout(0.2))
MergedModel.add(BatchNormalization())

MergedModel.add(Dense(1))
MergedModel.add(Activation('sigmoid'))

finModel = Model([model.input,model2.input,model3.input,model4.input,model5.input,model6.input],MergedModel(MergedModel1))
finModel.load_weights("weights.h5")
finModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

score = finModel.evaluate([X_test[:,0:40], X_test[:,40:80], X_test[:,0:40], X_test[:,40:80], X_test[:,0:40], X_test[:,40:80]],y_test)
print(score)
predictions = finModel.predict([X_test[:,0:40], X_test[:,40:80], X_test[:,0:40], X_test[:,40:80], X_test[:,0:40], X_test[:,40:80]]).ravel()
y_pred_class = binarize([predictions], 0.7)[0]
print(confusion_matrix(y_test, y_pred_class))
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, predictions)
auc_keras = auc(fpr_keras, tpr_keras)
print(auc_keras)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
