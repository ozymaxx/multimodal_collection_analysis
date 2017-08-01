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

TEST_PHRASE_FILE_NAME = "analysis_corpus/testphrases_our.txt"
TRAIN_PHRASE_FILE_NAME = "analysis_corpus/trainphrases_our.txt"
VAL_PHRASE_FILE_NAME = "analysis_corpus/valphrases_our.txt"
TRAIN_VAL_PHRASE_FILE_NAME = "analysis_corpus/trainvalphrases_our.txt"
EPOCH_COUNT = 5

VALIDATION = False

basevocab = {}
classes = {}

def constructConfusionMatrix(num_classes):
	result = []
	
	for i in range(num_classes):
		result.append([])
		
		for j in range(num_classes):
			result[i].append(0)
			
	return result
	
def rowSum(rowindex,confmatrix):
	return sum(confmatrix[rowindex])
	
def colSum(colindex,confmatrix):
	result = 0
	
	for i in range(len(confmatrix[0])):
		result = result + confmatrix[i][colindex]
		
	return result
	
def precision(confmatrix):
	result = [0]*len(confmatrix[0])
	
	for i in range(len(confmatrix[0])):
		if colSum(i,confmatrix) > 0:
			result[i] = confmatrix[i][i]*100.0/colSum(i,confmatrix)
		else:
			result[i] = 0
		
	return result
	
def recall(confmatrix):
	result = [0]*len(confmatrix[0])
	
	for i in range(len(confmatrix[0])):
		if rowSum(i,confmatrix) > 0:
			result[i] = confmatrix[i][i]*100.0/rowSum(i,confmatrix)
		else:
			result[i] = 0
		
	return result
	
def fm(precision,recall):
	result = [0]*len(precision)
	
	for i in range(len(precision)):
		if precision[i]+recall[i] > 0:
			result[i] = 2*precision[i]*recall[i]/(precision[i]+recall[i])
		else:
			result[i] = 0
		
	return result
	
def sumAll(confmatrix):
	result = 0
	
	for i in range(len(confmatrix[0])):
		result = result+sum(confmatrix[i])
		
	return result
	
def oneVsAll(confmatrix):
	onevsmatrix = [[0,0],[0,0]]
	
	for i in range(len(confmatrix[0])):
		rs = rowSum(i,confmatrix)
		cs = colSum(i,confmatrix)
		onevsmatrix[0][0] = onevsmatrix[0][0]+confmatrix[i][i]
		onevsmatrix[0][1] = onevsmatrix[0][1]+rs-confmatrix[i][i]
		onevsmatrix[1][0] = onevsmatrix[1][0]+cs-confmatrix[i][i]
		onevsmatrix[1][1] = onevsmatrix[1][1]+sumAll(confmatrix)-rs-cs+confmatrix[i][i]
		
	return onevsmatrix
	
def oneVsAllAcc(onevsmatrix):
	return 100.0*(onevsmatrix[0][0]+onevsmatrix[1][1])/(onevsmatrix[0][0]+onevsmatrix[0][1]+onevsmatrix[1][0]+onevsmatrix[1][1])
	
def loadCmatFromFile(file_path):
	cmat = []
	
	with open(file_path,'r') as cfile:
		for line in cfile:
			cmat.append([int(nm) for nm in line.split(' ')])
		cfile.close()
			
	return cmat
	
def saveCmatToFile(file_path,cmatrix):
	with open(file_path,'w') as cfile:
		for i in range(len(cmatrix[0])):
			for j in range(len(cmatrix[i])):
				cfile.write('%d' % cmatrix[i][j])
				
				if j < len(cmatrix[i])-1:
					cfile.write(' ')
			cfile.write('\n')
		cfile.close()

class CHARSCNN(nn.Module):
	def __init__(self,num_letters,num_classes,vocab_size,dwrd,kwrd,dchr,kchr,clu0,clu1,hlu):
		super(CHARSCNN,self).__init__()
		self.num_letters = num_letters
		self.num_classes = num_classes
		self.vocab_size = vocab_size
		self.dwrd = dwrd
		self.kwrd = kwrd
		self.dchr = dchr
		self.kchr = kchr
		self.clu0 = clu0
		self.clu1 = clu1
		self.hlu = hlu
		
		self.wchr = nn.Linear(num_letters,dchr,bias=False)
		self.wwrd = nn.Linear(vocab_size,dwrd,bias=False)
		self.w0 = nn.Linear(dchr*kchr,clu0)
		self.w1 = nn.Linear((dwrd+clu0)*kwrd,clu1)
		self.w2 = nn.Linear(clu1,hlu)
		self.w3 = nn.Linear(hlu,num_classes)
		
	def forward(self,sentence_charlevel,sentence_wordlevel):
		# compute word-level embeddings
		rwrds = [self.wwrd(v) for v in sentence_wordlevel]
		
		# compute character-level embeddings
		charembs = []
		
		for word_charlevel in sentence_charlevel:
			rchrs = [self.wchr(v) for v in word_charlevel]
			conved = []
			
			for charindex in range((self.kchr-1)/2,len(rchrs)-(self.kchr-1)/2):
				zm = rchrs[charindex-(self.kchr-1)/2]
				
				for i in range(1,self.kchr):
					zm = torch.cat((zm,rchrs[charindex-(self.kchr-1)/2+i]),1)
					
				#zm = torch.t(zm)
				#print zm
				conved.append(self.w0(zm))
				
			charemb = conved[0]
			
			for i in range(1,len(conved)):
				charemb = torch.max(charemb,conved[i])
				
			charembs.append(charemb)
		
		# compute sentence level representation
		wordvecs = []
		
		for i in range(len(sentence_wordlevel)):
			if i < (self.kwrd-1)/2:
				wordvecs.append(torch.cat((rwrds[i],autograd.Variable(torch.zeros(self.clu0).view(1,-1).cuda())),1))
			elif i < len(sentence_wordlevel)-(self.kwrd-1)/2:
				wordvecs.append(torch.cat((rwrds[i],charembs[i-(self.kwrd-1)/2]),1))
			else:
				wordvecs.append(torch.cat((rwrds[i],autograd.Variable(torch.zeros(self.clu0).view(1,-1).cuda())),1))
				
		wconved = []
		
		for wordindex in range((self.kwrd-1)/2,len(wordvecs)-(self.kwrd-1)/2):
			zm = wordvecs[wordindex-(self.kwrd-1)/2]
			
			for i in range(1,self.kwrd):
				zm = torch.cat((zm,wordvecs[wordindex-(self.kwrd-1)/2+i]),1)
				
			#zm = torch.t(zm)
			wconved.append(self.w1(zm))
			
		sentencemb = wconved[0]
		
		for i in range(1,len(wconved)):
			sentencemb = torch.max(sentencemb,wconved[i])
		
		# scoring
		return F.log_softmax(self.w3(F.tanh(self.w2(sentencemb))))
		
def make_wordlevel(vocabindices,sentence,kwrd):
	result = []
	
	for i in range((kwrd-1)/2):
		vec = [0]*(len(vocabindices.keys())+1)
		vec[len(vocabindices.keys())] = 1
		result.append(autograd.Variable(torch.FloatTensor(vec).view(1,-1).cuda()))
		
	words = sentence.split(' ')
	
	for word in words:
		vec = [0]*(len(vocabindices.keys())+1)
		vec[vocabindices[word]] = 1
		result.append(autograd.Variable(torch.FloatTensor(vec).view(1,-1).cuda()))
		
	for i in range((kwrd-1)/2):
		vec = [0]*(len(vocabindices.keys())+1)
		vec[len(vocabindices.keys())] = 1
		result.append(autograd.Variable(torch.FloatTensor(vec).view(1,-1).cuda()))
		
	return result
	
def make_charlevel(letterindices,sentence,kchr):
	result = []
	words = sentence.split(' ')
	
	#print '---'
	for word in words:
		word = word.strip()
		
		if len(word) > 0:
			vecs = []
			
			padding = ''
			
			for i in range((kchr-1)/2):
				padding = padding + '*'
				
			word = padding+word+padding
			word = unicode(word,'utf-8')
			
			#print word
			for chi in range(len(word)):
				if word[chi] in letterindices:
					vec = [0]*(len(letterindices.keys()))
					#print letterindices[word[chi]]
					vec[letterindices[word[chi]]] = 1
					vecs.append(autograd.Variable(torch.FloatTensor(vec).view(1,-1).cuda()))
				
			result.append(vecs)
		
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
print "Deep (char-based)"

letters = [u'a',u'b',u'c',u'ç',u'd',u'e',u'f',u'g',u'ğ',u'h',u'ı',u'i',u'j',u'k',u'l',u'm',u'n',u'o',u'ö',u'p',u'r',u's',u'ş',u't',u'u',u'ü',u'v',u'y',u'z',u'*',u'x',u'q']
#letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','*']
letterindices = {}

for i in range(len(letters)):
	letterindices[letters[i]] = i

vocabindices_withoutstems = {}
ind = 0

for line in testsentences:
	words = line.split(' ')
	
	for word in words:
		if not word in vocabindices_withoutstems:
			vocabindices_withoutstems[word] = ind
			ind = ind+1
			
for line in trainsentences:
	words = line.split(' ')
	
	for word in words:
		if not word in vocabindices_withoutstems:
			vocabindices_withoutstems[word] = ind
			ind = ind+1
		
if VALIDATION:	
	for line in valsentences:
		words = line.split(' ')
		
		for word in words:
			if not word in vocabindices_withoutstems:
				vocabindices_withoutstems[word] = ind
				ind = ind+1
			
num_classes = len(classes.keys())
num_letters = len(letters)
vocab_size = len(vocabindices_withoutstems.keys())+1
dwrd = 30
kwrd = 5
dchr = 5
kchr = 3
clu0 = 10
clu1 = 300
hlu = 300
	
model = None
optimizer = None
loss_function = None

model = None
valacc = 0

cond = False

if VALIDATION:
	cond = os.path.isfile('deep_charbased.model') and os.path.isfile('deep_charbased_val.acc')
else:
	cond = os.path.isfile('deep_charbased.model')

if cond:
	model = torch.load('deep_charbased.model')
	
	if VALIDATION:
		with open('deep_charbased_val.acc','r') as dval:
			for line in dval:
				valacc = float(line)
			dval.close()
else:
	model = CHARSCNN(num_letters,num_classes,vocab_size,dwrd,kwrd,dchr,kchr,clu0,clu1,hlu).cuda()
	optimizer = optim.SGD(model.parameters(),lr=0.02)
	loss_function = nn.NLLLoss()
	
	correctp = 0

	for epoch in range(EPOCH_COUNT):
		print "epoch %d" % epoch
		for i in range(len(trainvlab)):
			if len(trainsentences[i].split(' ')[0].strip()) > 0:
				model.zero_grad()
				target = autograd.Variable(makelabel_new(labindices,trainvlabstr[i]))
				logprobs = model(make_charlevel(letterindices,trainsentences[i],kchr),make_wordlevel(vocabindices_withoutstems,trainsentences[i],kwrd))
				loss = loss_function(logprobs,target)
				
				if i % 10000 == 0:
					print '%d instances have been iterated, with the loss given below' % i
					print loss
				
				loss.backward()
				optimizer.step()
	
	if VALIDATION:			
		cmat = constructConfusionMatrix(len(classes.keys()))
				
		for i in range(len(vvlab)):
			logprobs = model(make_charlevel(letterindices,valsentences[i],kchr),make_wordlevel(vocabindices_withoutstems,valsentences[i],kwrd))
			
			_,pred = logprobs.data.topk(1,1)
			cllab = pred.cpu().numpy().tolist()[0][0]
			
			cmat[vvlab[i]][int(cllab)] = cmat[vvlab[i]][int(cllab)]+1
				
		onevsallv = oneVsAll(cmat)
		valacc = oneVsAllAcc(onevsallv)
		
		print "One-vs-all validation accuracy = %.2f%%" % (valacc)
		
		with open('deep_charbased_val.acc','w') as dval:
			dval.write('%f\n' % valacc)
			dval.close()
	
	torch.save(model,'deep_charbased.model')
		
testtime = 0
		
if os.path.isfile('deep_charbased_confusion.matrix') and os.path.isfile('deep_charbased_prediction.time'):
	cmat = loadCmatFromFile('deep_charbased_confusion.matrix')
	
	with open('deep_charbased_prediction.time','r') as dtime:
		for line in dtime:
			testtime = float(line)
		dtime.close()
else:
	cmat = constructConfusionMatrix(len(classes.keys()))
	
	for i in range(len(testlab)):
		start_time_sub = time.time()
		logprobs = model(make_charlevel(letterindices,testsentences[i],kchr),make_wordlevel(vocabindices_withoutstems,testsentences[i],kwrd))
		end_time_sub = time.time()
		
		testtime = testtime + end_time_sub-start_time_sub
		
		_,pred = logprobs.data.topk(1,1)
		cllab = pred.cpu().numpy().tolist()[0][0]
		cmat[testlab[i]][int(cllab)] = cmat[testlab[i]][int(cllab)]+1
		
	saveCmatToFile('deep_charbased_confusion.matrix',cmat)
	
	with open('deep_charbased_prediction.time','w') as dtime:
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
