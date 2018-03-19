import numpy as np
import gensim
from gensim.models import word2vec,Word2Vec
import logging
from keras import regularizers
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, TimeDistributed, Bidirectional,GRU
from keras.layers.core import Activation, Dense, Dropout, Flatten, Lambda
from keras import regularizers,applications,optimizers,constraints
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout,Flatten,concatenate,Input,Reshape,dot,add,multiply,concatenate,Reshape
from keras.models import load_model
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers.embeddings import Embedding
from keras.models import Model, Sequential, load_model
from keras.optimizers import SGD, Adam
from keras import backend as K
import random
from PointerLSTM import PointerLSTM
from keras import utils
from keras.constraints import max_norm
from keras.utils.np_utils import to_categorical
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
np.set_printoptions(threshold=np.inf)




def load_data(context_path='new_train_context.npy',question_path='new_train_question.npy'):
	return np.load(context_path),np.load(question_path)

def word_embedding(data,mincount=3,n=256,load=True):
	if load:
		model = Word2Vec.load('skipgram_')

def get_dict_wordvec(train_context,max_length,n_comp,model_path='skipgram_'):
	def index_array(X,max_length,n_comp):
		return np.concatenate([[word2idx.get('_PAD') if word2idx.get(x) is None else word2idx.get(x) for x in X],np.zeros((max_length-len(X)))])
	model = Word2Vec.load(model_path)
	word2idx = {"_PAD": 0} 
	vocab_list = [(k, model.wv[k]) for k, v in model.wv.vocab.items()]
	embeddings_matrix = np.zeros((len(model.wv.vocab.items()) + 1, model.vector_size))
	for i in range(len(vocab_list)):
		word = vocab_list[i][0]
		word2idx[word] = i + 1
		embeddings_matrix[i + 1] = vocab_list[i][1]
	train_context_vec = [index_array(x,max_length,n_comp) for x in train_context] 
	return np.array(train_context_vec),embeddings_matrix






#Lambda fuction
def MM(inputlist):
	Q = inputlist[0]
	P = inputlist[1]
	QPT = K.batch_dot(Q,K.permute_dimensions(P,[0,2,1]))
	return QPT
def projection(inputlist):
	W = inputlist[0]
	Q= inputlist[1]
	return K.batch_dot(K.permute_dimensions(W,[0,2,1]),Q)
def rowsum(weightmatrix):
	return K.sum(weightmatrix,axis=1)
#Lambda fuction








def training(train_context_vec,train_question_vec,train_span,train_reg,n_comp,embeddings_matrix):
	checkpoint_entropy = ModelCheckpoint('./model/model_cross_entropy.h5', monitor='val_start_output_loss', verbose=1, save_best_only=True, mode='min')
	checkpoint_loss = ModelCheckpoint('./model/model_loss.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
	batchSize = 256
	LSTM_units = 64
	index = [i for i in range(len(train_context_vec))]
	random.shuffle(index)
	train_context_vec = train_context_vec[index]
	train_question_vec = train_question_vec[index]
	train_span = train_span[index]
	train_reg = train_reg[index]
	context_input = Input(shape=(train_context_vec.shape[1],))
	context_embedding = Embedding(len(embeddings_matrix),n_comp,weights=[embeddings_matrix],trainable=False,mask_zero=False)(context_input)
	context_LSTM = GRU(LSTM_units,return_sequences=True,dropout=0.0)(context_embedding)
	# context_LSTM = Dropout(0.5)(context_LSTM)

	question_input = Input(shape=(train_question_vec.shape[1],))
	question_embedding = Embedding(len(embeddings_matrix),n_comp,weights=[embeddings_matrix],trainable=False,mask_zero=False)(question_input)
	question_LSTM = GRU(LSTM_units,return_sequences=True,dropout=0.0)(question_embedding)
	# question_LSTM = Dropout(0.5)(question_LSTM)	
	
	
	weight_matrix = Lambda(MM)([question_LSTM,context_LSTM])
	WTQ = Lambda(projection)([weight_matrix,question_LSTM])

	WTQ_lstm = GRU(48,return_sequences=True,dropout=0.0)(WTQ)
	entropy_output1 = TimeDistributed(Dense(1,activation='tanh'))(Dropout(0.25)(WTQ_lstm))
	F1 = Flatten()(entropy_output1)
	entropy_output1 = Activation('softmax',name='start_output')(F1)


	WTQ_lstm2 = GRU(48,return_sequences=True,dropout=0.0)(WTQ)
	entropy_output2 = TimeDistributed(Dense(1,activation='tanh'))(Dropout(0.25)(WTQ_lstm2))
	F2 = Flatten()(entropy_output2)
	entropy_output2 = Activation('softmax',name='end_output')(F2)
	# model = load_model('./model/model_acc.h5')
	
	model = Model([context_input,question_input],[entropy_output1,entropy_output2])
	model.compile(loss={'start_output':'categorical_crossentropy','end_output':'categorical_crossentropy'}
		,optimizer='adam',metrics={'start_output':'acc','end_output':'acc'})
	his = model.fit([train_context_vec,train_question_vec], [train_span,train_reg],batch_size=batchSize,epochs=500,verbose=1,validation_split=0.05)#,callbacks=[checkpoint_loss,checkpoint_entropy,checkpoint_acc,checkpoint_acc2])







def prediction(model_path,context,question):
	model = load_model(model_path)
	model.summary()
	pred = model.predict([context,question],verbose=1, batch_size=128)
	# pd.DataFrame(pred).to_csv('./Prediction/pred.csv')
	print(pred[0])

if __name__ == '__main__':
	
	train_context,train_question = load_data()
	train_span=np.load('new_train_span.npy')
	train_span=np.array(train_span)
	train_start=train_span[:,0]
	train_end=train_span[:,1]
	train_start=utils.to_categorical(train_start,264)
	train_end=utils.to_categorical(train_end,264)
	# train_start=np.reshape(train_start,(11411,1,264))
	# train_end=np.reshape(train_end,(11411,1,264))
	# y=np.concatenate((train_start,train_end),1)
	data = np.concatenate([train_context,train_question])

	n_comp = 256
	max_length_question = int(np.percentile([len(x) for x in train_question],100))
	max_length_context = int(np.percentile([len(x) for x in train_context],100))
	print(max_length_question,max_length_context)
	word_embedding(data,mincount=3,n=n_comp,load=True)
	train_context_vec, embeddings_matrix = get_dict_wordvec(train_context,max_length_context,n_comp)
	train_question_vec,embeddings_matrix = get_dict_wordvec(train_question,max_length_question,n_comp)
	print(train_context_vec.shape,train_question_vec.shape)
	training(train_context_vec,train_question_vec,train_start,train_end,n_comp,embeddings_matrix)
	







	# prediction('./model/model_loss.h5',train_context_vec,train_question_vec)

	# print(np.percentile([len(x) for x in train_context],100))
	# print(np.percentile([len(x) for x in train_question],100))
	# [print(x) for x in train_question if len(x)==3]