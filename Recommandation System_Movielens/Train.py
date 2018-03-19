import pandas 
import os
import requests
from sklearn import dummy, metrics, cross_validation, ensemble
import numpy as np
import keras.models as kmodels
import keras.layers as klayers
import keras.backend as K
import keras
from keras.layers import Input,Embedding,Flatten,Dot,Add
from keras.constraints import max_norm
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, load_model
import csv
from keras import regularizers
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from keras.models import Model



def  readdata():
	users = pandas.read_csv('users.csv', sep='::', engine='python')

	movies = pandas.read_csv('movies.csv', engine='python',sep='::')
	Genres = movies.Genres.str.split('|',expand=True)
	movies=pandas.concat([movies,Genres], axis=1)
	ratings = pandas.read_csv('train.csv' )
	
	return ratings,movies,users





def ta_mf_model(n_items,n_users,latent_dim = 256):
	user_input = Input(shape = [1])
	item_input = Input(shape = [1])
	user_vec = Embedding(n_users,latent_dim,embeddings_initializer = 'random_normal')(user_input)
	user_vec = Flatten()(user_vec)
	item_vec = Embedding(n_items+1,latent_dim,embeddings_initializer = 'random_normal')(item_input)
	item_vec = Flatten()(item_vec)
	#user_bias = Embedding(n_users,1,embeddings_initializer = 'uniform')(user_input)
	#user_bias = Flatten()(user_bias)
	#item_bias = Embedding(n_items,1,embeddings_initializer = 'uniform')(item_input)
	#item_bias = Flatten()(item_bias)
	r_hat = Dot(axes = 1)([item_vec,user_vec])
	#r_hat = Add()([r_hat,item_bias,user_bias])
	r_hat = keras.layers.Dense(1,kernel_regularizer=regularizers.l2(10))(r_hat)
	model = keras.models.Model([item_input,user_input],r_hat)
	model.compile(loss = 'mse',optimizer = 'adam',metrics=['mse'])
	return(model)



def ccmodel (n_items,n_users,latent_dim = 1024):
	user_input = Input(shape = [1])
	item_input = Input(shape = [1])
	user_vec = Embedding(n_users,latent_dim,embeddings_initializer = 'random_normal')(user_input)
	user_vec = Flatten()(user_vec)
	item_vec = Embedding(n_items+1,latent_dim,embeddings_initializer = 'random_normal')(item_input)
	item_vec = Flatten()(item_vec)
	user_bias = Embedding(n_users,1,embeddings_initializer = 'uniform')(user_input)
	user_bias = Flatten()(user_bias)
	item_bias = Embedding(n_items,1,embeddings_initializer = 'uniform')(item_input)
	item_bias = Flatten()(item_bias)
	r_hat = Dot(axes = 1)([item_vec,user_vec])
	r_hat = Add()([r_hat,item_bias,user_bias])



	movie_vec = keras.layers.Flatten()(keras.layers.Embedding(n_items + 1, 256,embeddings_initializer = 'random_normal')(item_input))
	user_vec = keras.layers.Flatten()(keras.layers.Embedding(n_users, 256,embeddings_initializer = 'random_normal')(user_input))
	input_vecs = keras.layers.merge([movie_vec, user_vec,r_hat], mode='concat')


	input_vecs =( keras.layers.Dense(256,activation='relu',kernel_regularizer=regularizers.l2(4),kernel_constraint=max_norm(6.))(input_vecs))
	input_vecs = keras.layers.Dropout(0.5)(input_vecs)
	r_hat = keras.layers.Dense(1,kernel_regularizer=regularizers.l2(10),kernel_constraint=max_norm(5.))(input_vecs)
	model = keras.models.Model([item_input,user_input],r_hat)
	model.compile(loss = 'mse',optimizer = 'adam',metrics=['mse'])
	return(model)





def nn(n_movies,n_users):#320.2validate 0.7515~0.7580...
	movie_input = keras.layers.Input(shape=[1])
	movie_vec = keras.layers.Flatten()(keras.layers.Embedding(n_movies + 1, 32,embeddings_initializer = 'random_normal')(movie_input))
	movie_vec = keras.layers.Dropout(0.2)(movie_vec)


	user_input = keras.layers.Input(shape=[1])
	user_vec = keras.layers.Flatten()(keras.layers.Embedding(n_users + 1, 32,embeddings_initializer = 'random_normal')(user_input))
	user_vec = keras.layers.Dropout(0.2)(user_vec)

	input_vecs = keras.layers.merge([movie_vec, user_vec], mode='concat')
	nn = keras.layers.Dropout(0.5)(keras.layers.Dense(128, activation='relu',kernel_constraint=max_norm(2.))(input_vecs))
	nn = keras.layers.normalization.BatchNormalization()(nn)
	nn = keras.layers.Dropout(0.5)(keras.layers.Dense(128, activation='relu',kernel_constraint=max_norm(2.))(nn))
	nn = keras.layers.normalization.BatchNormalization()(nn)
	nn = keras.layers.Dropout(0.5)(keras.layers.Dense(128, activation='relu',kernel_constraint=max_norm(2.))(nn))
	result = keras.layers.Dense(1,kernel_regularizer=regularizers.l2(0.01))(nn)
	adamax=keras.optimizers.Adamax(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	model = kmodels.Model([movie_input, user_input], result)
	model.compile(adamax, 'mean_squared_error',metrics=['mse'])

	return model



def new_model(n_items,n_users,latent_dim = 64):
	user_input = Input(shape = [1])
	item_input = Input(shape = [1])
	ifo_input=Input(shape = [21])

	user_vec = Embedding(n_users,latent_dim,embeddings_initializer = 'random_normal')(user_input)
	user_vec = Flatten()(user_vec)
	item_vec = Embedding(n_items+1,latent_dim,embeddings_initializer = 'random_normal')(item_input)
	item_vec = Flatten()(item_vec)
	user_bias = Embedding(n_users,1,embeddings_initializer = 'uniform')(user_input)
	user_bias = Flatten()(user_bias)
	item_bias = Embedding(n_items+1,1,embeddings_initializer = 'uniform')(item_input)
	item_bias = Flatten()(item_bias)
	r_hat = Dot(axes = 1)([item_vec,user_vec])
	r_hat = Add()([r_hat,item_bias,user_bias])


	inputt =keras.layers.Dropout(0.5)(keras.layers.Dense(16, activation='relu',kernel_regularizer=regularizers.l2(0.1))(ifo_input))
	inputt = keras.layers.normalization.BatchNormalization()(inputt)
	inputt =keras.layers.Dropout(0.5)(keras.layers.Dense(8, activation='relu',kernel_regularizer=regularizers.l2(0.1))(inputt))

	input_vecs = keras.layers.merge([r_hat, inputt], mode='concat')
	result = (keras.layers.Dense(1,kernel_regularizer=regularizers.l2(1))(input_vecs))

	#adamax=keras.optimizers.Adamax(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	model = keras.models.Model([item_input,user_input,ifo_input],result)
	model.compile(loss = 'mse',optimizer = 'adam',metrics=['mse'])
	return(model)




def feature_add(ratings,movies,users):
	conditions = [(users['Age']<=18),(18<users['Age'])&(users['Age']<=30),(30<users['Age'])&(users['Age']<=50)]
	choices = ['young', 'strong', 'middle']
	users['age'] = np.select(conditions, choices, default='old')
	users=pandas.concat([users,pandas.get_dummies(users['age'],drop_first=True),pandas.get_dummies(users['Gender'],drop_first=True),pandas.get_dummies(users['Occupation'])], axis=1)
	apend=pandas.get_dummies(movies[0])
	for i in range(0,3883) :
		for j in range(4,9) :
			if movies.iloc[i,j] is not None :
				apend[movies.iloc[i,j]].iloc[i]+=1
	movies=pandas.concat([movies,apend], axis=1)
	return ratings,movies,users


def  metadata(ratings,movies,users):
	ratings=ratings.sample(frac=1)
	movieid = np.array(ratings.MovieID)
	userid = np.array(ratings.UserID)
	y = np.array(ratings.Rating)
	movies=movies.set_index('movieID')
	users=users.set_index('UserID')
	movie_ifo=movies.ix[movieid,9:28]
	user_ifo=users.ix[userid,5:9]
	movie_ifo=movie_ifo.values
	user_ifo=user_ifo.values
	return movieid,userid,y,movie_ifo,user_ifo



def draw(model_path='mse_model.h5'):
	model = load_model(model_path)
	model.summary()
	movie_emb=np.array(model.layers[2].get_weights()).squeeze()
	return movie_emb,model






def main():
	ratings,movies,users=readdata()
	ratings,movies,users=feature_add(ratings,movies,users)
	n_movies = np.max(ratings['MovieID'])
	n_users = np.max(ratings['UserID'])
	movieid,userid,y,movie_ifo,user_ifo=metadata(ratings,movies,users)
	ifo=np.concatenate((movie_ifo, user_ifo), axis=1)
	print(ifo.shape)



	model=new_model(n_movies,n_users)
	# checkpoint = ModelCheckpoint("bias_model.h5", monitor='val_mean_squared_error', verbose=1, save_best_only=True, mode='min')
	# model.fit([movieid, userid],y,verbose=1,batch_size=8000,epochs=1000,validation_split=0.05,callbacks=[checkpoint])
	model.summary()
	







#----------------------draw----------------------------------------------------------------
	# movie_emb,model=draw()
	# intermediate_layer_model = Model(inputs=model.input,outputs=model.layers[2].output)
	# out=intermediate_layer_model.predict([movieid[1:3],userid[1:3]])
	# vaild_emb=movie_emb[movies.movieID,:]

	# movies[0],cats=pandas.factorize(movies[0])
	# movies[0]=pandas.Categorical(movies[0], categories=np.arange(len(cats)))
	# y=np.array(movies[0])
	# print(vaild_emb.shape)


	# vis_data=TSNE(n_components=2).fit_transform(vaild_emb)
	# vis_x=vis_data[:,0]
	# vis_y=vis_data[:,1]

	# cm=plt.cm.get_cmap('RdYlBu')
	# sc=plt.scatter(vis_x,vis_y,c=y,cmap=cm)
	# plt.colorbar(sc)
	# plt.show()
#----------------------draw----------------------------------------------------------------






if __name__ == '__main__':
	main()
