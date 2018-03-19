import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, TimeDistributed, Bidirectional
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers.embeddings import Embedding
from keras.models import Model, Sequential, load_model
from keras.optimizers import SGD, Adam
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import re
import itertools
from gensim.models import word2vec
import logging

max_length=36
EMBEDDING_DIM =256
training_label = sys.argv[1]    
training_nolabel = sys.argv[2]  


def load_to_embed(readnpy=False):
	if readnpy:
		y_train = np.load('./feature/y_train.npy')
		train_text = np.load('./feature/train_text.npy')
		test_text = np.load('./feature/test_text.npy')
		embed_text = np.load('./feature/embed_text.npy')
	else:
		pattern = re.compile('[^A-Za-z0-9 -]')
		y_train = []
		train_text = []
		test_text = []
		embed_text=[]
		with open(training_label, "U", encoding='utf-8-sig') as f:
			for l in f:
				strs = re.sub("\d+", "", l.strip().split("+++$+++")[1])
				y_train.append(l.strip().split("+++$+++")[0])
				train_text.append(pattern.sub('',strs))
				embed_text.append(pattern.sub('',strs))
			np.save('./feature/y_train.npy', y_train)
			np.save('./feature/train_text.npy', train_text)
			
		with open(training_nolabel,"U", encoding='utf-8-sig') as f:
			for l in f:
				strs = re.sub("\d+", "", l.strip())
				embed_text.append(pattern.sub('',strs))


	return embed_text,y_train, train_text


def word22vec(sentence,switch=False):
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	
	if switch:
		model=word2vec.Word2Vec.load('mymodel')
	else:
		sentence=[s.split() for s in sentence]
		model = word2vec.Word2Vec(sentence, min_count=7,workers=8,size=256,iter=7,sg=1)
		model.save('mymodel')
	return model


def data_process(model,train):
	word2idx = {"_PAD": 0}
	vocab_list = [(k, model.wv[k]) for k, v in model.wv.vocab.items()]
	embeddings_matrix = np.zeros((len(model.wv.vocab.items()) + 1, model.vector_size))
	
	for i in range(len(vocab_list)):
		word = vocab_list[i][0]
		word2idx[word] = i + 1
		embeddings_matrix[i + 1] = vocab_list[i][1]
	return word2idx,embeddings_matrix


def create_idx_data(word2idx,train,switch=False):
	
	if switch:
		train_idx= np.load('./feature/train_idx.npy')
	else:
		train=[s.split() for s in train]
		train_idx=np.zeros((len(train), max_length), dtype='float32')
		for i in range(0,len(train)):
			for j in range(0,len(train[i])):
				try:
					train_idx[i][j]=word2idx[train[i][j]]
				except KeyError:
					train_idx[i][j]=0
		
		np.save('./feature/train_idx.npy', train_idx)
	

	return train_idx
		

def rnn(embeddings_matrix):
	model = Sequential()
	embedding_layer = Embedding(input_dim=len(embeddings_matrix),
                            output_dim=EMBEDDING_DIM,
                            input_length=max_length,
                            weights=[embeddings_matrix],
                            trainable=False)
	model.add(embedding_layer)
	model.add(LSTM(256))
	model.add(Dense(units=256, activation='relu'))
	model.add(Dense(units=1, activation='sigmoid'))
	model.compile(loss = 'binary_crossentropy',optimizer=Adam(), metrics=['accuracy'])
	return model


def prediction(test_X,model_path='best_model.h5'):
	model = load_model(model_path)
	pred = [1 if x[0] >= 0.5 else 0 for x in model.predict(test_X ,batch_size=512, verbose=1)]
	text = open('predict.csv', "w+")
	s = csv.writer(text, delimiter=',', lineterminator='\n')
	s.writerow(["id", "label"])
	[s.writerow([i,pred[i]]) for i in range(len(pred))]
	text.close()

def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc = 'upper left')
    plt.show()


def main():
	e,y,x=load_to_embed()
	mm=word22vec(e)
	print(mm.most_similar(['like']))
	print(mm.most_similar(['shut']))
	word2idx,embeddings_matrix=data_process(mm,x)
	train_idx=create_idx_data(word2idx,x)


	checkpoint = ModelCheckpoint("Try_model.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	model=rnn(embeddings_matrix)
	model.summary()
	train_history=model.fit(train_idx,y,verbose=1,batch_size=4096,epochs=30,validation_split=0.1)#,callbacks=["checkpoint"])



if __name__ == '__main__':
	main()



