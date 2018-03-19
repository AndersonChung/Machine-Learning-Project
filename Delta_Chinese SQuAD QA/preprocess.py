import jieba
import pandas as pd
import numpy as np
import re
from gensim.models import word2vec,Word2Vec


test_context = pd.read_csv('./data/testl.context', sep='delimiter', header=None, encoding='utf-8')
train_context = pd.read_csv('./data/train.context', sep='delimiter', header=None, encoding='utf-8')
test_question = pd.read_csv('./data/test.question', sep='delimiter', header=None, encoding='utf-8')
train_question = pd.read_csv('./data/train.question', sep='delimiter', header=None, encoding='utf-8')



def contex_to_charcter(data,path):
	data=data.as_matrix()
	data=np.reshape(data,data.shape[0])
	data_sentence=[]
	data_jiba=[]
	for  sentence in data :
		a=re.split('[。，]', sentence)
		a=a[0:len(a)-1]
		data_sentence.append(a)

	total = len(data_sentence)
	i = 1
	for  par in data_sentence :
		i+=1
		print("{0:.2f}".format(i / total))
		data_charcter=[]
		for sentence  in par:
			seg_list = jieba.cut(sentence)
			seg_list = ("|".join(seg_list)).split('|')
			data_charcter.append(seg_list)
		data_jiba.append(data_charcter)
	np.save(path, data_jiba)


def question_to_charcter(data,path):
	data=data.as_matrix()
	data=np.reshape(data,data.shape[0])
	data_jiba=[]
	for  sentence in data :
		seg_list = jieba.cut(sentence)
		seg_list = ("|".join(seg_list)).split('|')
		seg_list=seg_list[0:-1]
		data_jiba.append(seg_list)
	np.save(path, data_jiba)

# question_to_charcter(test_question,"test_question")
# question_to_charcter(train_question,"train_question")



#def simimlarity(jiba,question):



test_question=np.load("test_question.npy")
train_question=np.load("train_question.npy")
test_jiba=np.load("test_jiba.npy")
train_jiba=np.load("train_jiba.npy")
model = Word2Vec.load('skipgram_')









# par_weight=[]

# for i in range(len(test_jiba)):
# 	print(str(i/len(test_jiba)))
# 	question=[]
# 	test_question[i]=test_question[i]
# 	for char in test_question[i]:
# 		try:
# 			question.append(np.array(model.wv[char]))
# 		except KeyError:
# 			question.append(np.zeros(256))
# 	question=np.array(question)
# 	question=np.divide(question,np.linalg.norm(question,axis=0))
# 	sentence_weight=[]
# 	for sentence in test_jiba[i]:
# 		char_weight=[]
# 		for char in sentence:
# 			try:
# 				a=np.array(model.wv[char])
# 			except KeyError:
# 				a=np.zeros(256)
# 			weight=np.dot(question,np.reshape(a/np.linalg.norm(a),(256,1)))
# 			char_weight.append(weight)
# 		char_weight=np.array(char_weight)
# 		sentence_weight.append(np.average(np.amax(char_weight, axis=1)))
# 	par_weight.append(sentence_weight)

#for i in range(len(par_weight)):
	#par_weight[i]=np.nan_to_num(np.array(par_weight[i]))
# np.save("par_weight_test",par_weight)





par_weight=np.load("par_weight_test.npy")

# weight=[]
# for a in par_weight:
# 	weight.append(np.argmax(a))
# np.save("weight_test",weight)
weight=np.load("weight_test.npy")
weight=np.array(weight)


# def find_contex_ind(weight,data):
# 	data=data.as_matrix()
# 	data=np.reshape(data,data.shape[0])
# 	index=[]
# 	for i in range(data.shape[0]):
# 		mylist=list(str(data[i]))
# 		a=[j for j,x in enumerate(mylist) if x =='，']
# 		b=[j for j,x in enumerate(mylist) if x =='。']
# 		c=a+b
# 		c.sort()
# 		front=weight[i]-5
# 		back=weight[i]+5
# 		if back>=len(c):
# 			back=len(c)-1
# 		if front<=0:
# 			front=0
# 			index.append([0,c[back]-1])
# 		else:
# 			index.append([c[front-1]+1,c[back]-1])
# 	np.save("index_test",index)


# find_contex_ind(weight,test_context)
index=np.load("index_test.npy")


train_span = pd.read_csv('./data/train.span', sep='delimiter', header=None, encoding='utf-8')
train_span_save = np.array([x[0].split(' ') for x in train_span.as_matrix()]).astype('int')


# def split_context(data,index,span,path1,path2):
# 	data=data.as_matrix()
# 	data=np.reshape(data,data.shape[0])
# 	da=[]
# 	filt=[]
# 	for i in range(data.shape[0]):
# 		#if span[i][0]>=index[i][0] and span[i][1]<=index[i][1]:
# 		l=list(data[i])[index[i][0]:index[i][1]+1]
# 		da.append(''.join(str(e) for e in l))
# 		filt.append(i)
# 	data_jiba=[]
# 	for  sentence in da :
# 		seg_list = jieba.cut(sentence)
# 		seg_list = ("|".join(seg_list)).split('|')
# 		data_jiba.append(seg_list)
# 	np.save(path1, data_jiba)
# 	#np.save(path2, filt)
# split_context(test_context,index,train_span_save,"new_test_context","filt")
new_test_context=np.load("new_test_context.npy")
# filt=np.load("filt.npy")


# def new_span(data,index,span,path):
# 	new=[]
# 	for i in range(len(data)):
# 		if span[i][0]>=index[i][0] and span[i][1]<=index[i][1]:
# 			sum=0
# 			for j in range(len(data[i])):
# 				if(index[i][0]+sum>span[i][0]):
# 					start=j-1
# 					break
# 				elif(index[i][0]+sum==span[i][0]):
# 					start=j
# 					break
# 				else:
# 					sum+=len(data[i][j])
# 			sum=0
# 			for j in range(len(data[i])):
# 				if(index[i][0]+sum>span[i][1]):
# 					end=j-1
# 					break
# 				elif(index[i][0]+sum==span[i][1]):
# 					end=j
# 					break
# 				else:
# 					sum+=len(data[i][j])
# 			new.append([start,end])
# 			np.save(path, new)

# #new_span(new_train_context,index[filt],train_span_save[filt],"new_train_span")
# new_train_span=np.load("new_train_span.npy")
# #new_train_question=train_question[filt]
# new_train_question=np.load("new_train_question.npy")






# def recon(data,ind,out):
# 	out=np.array(out)
# 	start=out[:,0]
# 	end=out[:,1]
# 	re_start=[]
# 	re_end=[]
# 	for i in range(len(start)):
# 		sum=ind[i][0]
# 		for j in range(start[i]):
# 			sum+=len(data[i][j])
# 		re_start.append(sum)
# 		sum=0
# 		for j in range(end[i]+1):
# 			if j==0:
# 				sum+=ind[i][0]+len(data[i][j])-1
# 			elif j>=len(data[i]) :
# 				sum=sum
# 			else:
# 				sum+=len(data[i][j])
# 		re_end.append(sum)
# 	re_end=np.array(re_end)
# 	re_start=np.array(re_start)
# 	return re_start,re_end

index=np.load("index.npy")
filt=np.load("filt.npy")
np.save("index_train",index[filt])
# new_train_context=np.load("new_train_context.npy")

# new_train_span=np.load("new_train_span.npy")
# re_start,re_end=recon(new_train_context,index[filt],new_train_span)
# print(re_start[30:50])
# train_span_save=train_span_save[filt]
# print(train_span_save[30:50])