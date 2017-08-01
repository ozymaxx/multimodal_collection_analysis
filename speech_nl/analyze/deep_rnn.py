#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import random
import sys
import time
import numpy as np
import os

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gensim.models import Word2Vec

from confmatrix import *

VALIDATION = False

TEST_PHRASE_FILE_NAME = "analysis_corpus/testphrases_our.txt"
TRAIN_PHRASE_FILE_NAME = "analysis_corpus/trainphrases_our.txt"
VAL_PHRASE_FILE_NAME = "analysis_corpus/valphrases_our.txt"
TRAIN_VAL_PHRASE_FILE_NAME = "analysis_corpus/trainvalphrases_our.txt"
STEMS_FILE_NAME = "stems_rej.txt"
EPOCH_COUNT = 5

basevocab = {}
classes = {}

totalwords = 0
wordrejected = 0

def produceW2V(word,w2vmodel,vsize):
	global wordrejected
	global totalwords
	
	result = None
	totalwords = totalwords + 1
	
	try:
		result = w2vmodel.wv[word]
	except KeyError:
		wordrejected = wordrejected+1
		result = [0]*vsize
		result[vsize-1] = 1
		result = np.array(result)
		
	return result

class CNN(nn.Module):
	def __init__(self,n,h,m,p,k):
		super(CNN,self).__init__()
		self.n = n
		self.h = h
		self.m = m
		self.p = p
		self.k = k
		
		self.conv = nn.Linear(m*h,n)
		self.dout = nn.Dropout(p=p)
		self.ff1 = nn.Linear(n,k)
		self.ff2 = nn.Linear(k,k)
		
	def forward(self,sentence_wordlevel):
		# convolution
		conved = []
		
		for i in range(len(sentence_wordlevel)-self.h+1):
			zm = sentence_wordlevel[i]
			
			for j in range(1,self.h):
				zm = torch.cat((zm,sentence_wordlevel[i+j]),1)
				
			#zm = torch.t(zm)
			#print zm
			conved.append(F.relu(self.conv(zm)))
			
		# max-pooling for sentence representation
		sentencemb = conved[0]
		
		for i in range(1,len(conved)):
			sentencemb = torch.max(sentencemb,conved[i])
			
		# drop-out after sentence representation
		dropped = self.dout(sentencemb)
		
		# 2 more feed-forward layers and scoring
		return F.log_softmax(self.ff2(F.tanh(self.ff1(dropped))))
		
def make_wordlevel(sentence,w2vmodel,vsize,convsize):
	result = []
	
	words = sentence.split(' ')
	
	if len(words) >= convsize:
		for word in words:
			result.append(autograd.Variable(torch.FloatTensor(produceW2V(word,w2vmodel,vsize)).view(1,-1).cuda()))
	else:
		prelen = int(math.ceil(float(convsize-len(words))/2))
		trlen = convsize-prelen-len(words)
		
		for i in range(prelen):
			arr = [0]*vsize
			arr[0] = 1
			result.append(autograd.Variable(torch.FloatTensor(arr).view(1,-1).cuda()))
			
		for word in words:
			result.append(autograd.Variable(torch.FloatTensor(produceW2V(unicode(word,'utf-8'),w2vmodel,vsize)).view(1,-1).cuda()))
			
		for i in range(trlen):
			arr = [0]*vsize
			arr[0] = 1
			result.append(autograd.Variable(torch.FloatTensor(arr).view(1,-1).cuda()))
		
	return result
	
def makelabel_new(labindices,label):
	return torch.LongTensor([labindices[label]]).cuda()
	
with open(TRAIN_PHRASE_FILE_NAME,'r') as phrasefile:
	for line in phrasefile:
		tokens = line.split('-')
		label = tokens[1].strip()
			
		if not label in classes:
			classes[label] = 1
		else:
			classes[label] = classes[label]+1
			
	phrasefile.close()
	
labkeys = classes.keys()
labkeys.sort()
labindices = {}

for i in range(len(labkeys)):
	labindices[labkeys[i]] = i
	
testsentences = []
trainsentences = []
valsentences = []

labstrs = []

testlab = []
testlabstr = []
trainvlab = []
trainvlabstr = []
vvlab = []
vvlabstr = []

totalwords = 0

with open(TEST_PHRASE_FILE_NAME,'r') as testphrases:
	for line in testphrases:
		sides = line.split('-')
		sides = [s.strip() for s in sides]
		totalwords = totalwords + len(sides[0].split(' '))
		testlabstr.append(sides[1])
		testlab.append(labindices[sides[1]])       
		testsentences.append(sides[0])
		
	testphrases.close()
	
if VALIDATION:	
	with open(VAL_PHRASE_FILE_NAME,'r') as valphrases:
		for line in valphrases:
			sides = line.split('-')
			sides = [s.strip() for s in sides]
			totalwords = totalwords + len(sides[0].split(' '))
			vvlab.append(labindices[sides[1]])
			vvlabstr.append(sides[1])
			valsentences.append(sides[0])
			
		valphrases.close()
		
	with open(TRAIN_PHRASE_FILE_NAME,'r') as trainphrases:
		for line in trainphrases:
			sides = line.split('-')
			sides = [s.strip() for s in sides]
			totalwords = totalwords + len(sides[0].split(' '))
			trainvlab.append(labindices[sides[1]])
			trainvlabstr.append(sides[1])
			trainsentences.append(sides[0])
			
		trainphrases.close()
else:
	with open(TRAIN_VAL_PHRASE_FILE_NAME,'r') as trainphrases:
		for line in trainphrases:
			sides = line.split('-')
			sides = [s.strip() for s in sides]
			totalwords = totalwords + len(sides[0].split(' '))
			trainvlab.append(labindices[sides[1]])
			trainvlabstr.append(sides[1])
			trainsentences.append(sides[0])
			
		trainphrases.close()

print ""
print "Deep (RNN)"

w2vmodel = Word2Vec.load('tr.bin')
			
num_classes = len(classes.keys())
n = 500
h = 3
m = 200
p = 0.5
	
model = None
optimizer = None
loss_function = None

model = None
valacc = 0

cond = False

if VALIDATION:
	cond = os.path.isfile('deep_rnn.model') and os.path.isfile('deep_rnn_val.acc')
else:
	cond = os.path.isfile('deep_rnn.model')

if cond:
	model = torch.load('deep_rnn.model')
	
	if VALIDATION:
		with open('deep_rnn_val.acc','r') as dval:
			for line in dval:
				valacc = float(line)
			dval.close()
else:
	model = CNN(n,h,m,p,num_classes).cuda()
	optimizer = optim.Adadelta(model.parameters())
	loss_function = nn.NLLLoss()
	
	correctp = 0

	for epoch in range(EPOCH_COUNT):
		print "epoch %d" % epoch
		for i in range(len(trainvlab)):
			if len(trainsentences[i].split(' ')[0].strip()) > 0:
				model.zero_grad()
				target = autograd.Variable(makelabel_new(labindices,trainvlabstr[i]))
				logprobs = model(make_wordlevel(trainsentences[i],w2vmodel,m,h))
				loss = loss_function(logprobs,target)
				
				if i % 10000 == 0:
					print '%d instances have been iterated, with the loss given below' % i
					print loss
				
				loss.backward()
				optimizer.step()
				
	cmat = constructConfusionMatrix(len(classes.keys()))
			
	if VALIDATION:
		for i in range(len(vvlab)):
			logprobs = model(make_wordlevel(valsentences[i],w2vmodel,m,h))
			
			_,pred = logprobs.data.topk(1,1)
			cllab = pred.cpu().numpy().tolist()[0][0]
			
			cmat[vvlab[i]][int(cllab)] = cmat[vvlab[i]][int(cllab)]+1
				
		onevsallv = oneVsAll(cmat)
		valacc = oneVsAllAcc(onevsallv)
		
		print "One-vs-all validation accuracy = %.2f%%" % (valacc)
		
		with open('deep_rnn_val.acc','w') as dval:
			dval.write('%f\n' % valacc)
			dval.close()
	
	torch.save(model,'deep_rnn.model')
		
testtime = 0
		
if os.path.isfile('deep_rnn_confusion.matrix') and os.path.isfile('deep_rnn_prediction.time'):
	cmat = loadCmatFromFile('deep_rnn_confusion.matrix')
	
	with open('deep_rnn_prediction.time','r') as dtime:
		for line in dtime:
			testtime = float(line)
		dtime.close()
else:
	cmat = constructConfusionMatrix(len(classes.keys()))
	
	for i in range(len(testlab)):
		start_time_sub = time.time()
		logprobs = model(make_wordlevel(testsentences[i],w2vmodel,m,h))
		end_time_sub = time.time()
		
		testtime = testtime + end_time_sub-start_time_sub
		
		_,pred = logprobs.data.topk(1,1)
		cllab = pred.cpu().numpy().tolist()[0][0]
		cmat[testlab[i]][int(cllab)] = cmat[testlab[i]][int(cllab)]+1
		
	saveCmatToFile('deep_rnn_confusion.matrix',cmat)
	
	with open('deep_rnn_prediction.time','w') as dtime:
		dtime.write('%f\n' % testtime)
		dtime.close()


print "# of test instances = %d" % len(testlab) 
print "Avg. classification time of an instance = %.2f seconds" % (testtime/len(testlab))

onevsall = oneVsAll(cmat)

print "One-vs-all accuracy = %.2f%%" % oneVsAllAcc(onevsall)
print ""

print "Precision rates for each class"
precisions = precision(cmat)
clsss = classes.keys()

for i in range(len(clsss)):
	print "%s = %.2f%%" % (labkeys[i],precisions[i])
	
print "Recall rates for each class"
recalls = recall(cmat)
clsss = classes.keys()

for i in range(len(clsss)):
	print "%s = %.2f%%" % (labkeys[i],recalls[i])
	
print "F1 rates for each class"
fs = fm(precisions,recalls)
clsss = classes.keys()

for i in range(len(clsss)):
	print "%s = %.2f%%" % (labkeys[i],fs[i])
	
print ""
print "wordrejected = %d" % wordrejected
print "totalwords = %d" % totalwords
