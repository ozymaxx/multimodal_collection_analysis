#!/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn.neighbors import KNeighborsClassifier
import math
import random
import sys
import time
import numpy as np
import os
import cPickle

from confmatrix import *

VALIDATION = False

TEST_PHRASE_FILE_NAME = "analysis_corpus/testphrases_our.txt"
TRAIN_PHRASE_FILE_NAME = "analysis_corpus/trainphrases_our.txt"
VAL_PHRASE_FILE_NAME = "analysis_corpus/valphrases_our.txt"
TRAIN_VAL_PHRASE_FILE_NAME = "analysis_corpus/trainvalphrases_our.txt"
STEMS_FILE_NAME = "stems_all.txt"
ZEMBEREK_STEMMING = True
STEM_LENGTH = 3

stems = {}
basevocab = {}
classes = {}

def makebow(vocabindices,words):
	result = [0]*len(vocabindices.keys())
	
	for word in words:
		result[vocabindices[word]] = result[vocabindices[word]]+1
		
	return result
	
with open(STEMS_FILE_NAME,'r') as stemsfile:
	for line in stemsfile: 
		tokens = line.split(" ")
		tokens = [t.strip() for t in tokens]
		
		if ZEMBEREK_STEMMING:
			if not tokens[0] in stems:
				stems[tokens[0]] = tokens[1]
		else:
			if not tokens[0] in stems:
				stems[tokens[0]] = tokens[0][0:STEM_LENGTH]
				
	stemsfile.close()

for cls in stems.keys():
	if not stems[cls] in basevocab:
		basevocab[stems[cls]] = True

with open(TRAIN_PHRASE_FILE_NAME,'r') as phrasefile:
	for line in phrasefile:
		tokens = line.split('-')
		label = tokens[1].strip()
			
		if not label in classes:
			classes[label] = 1
		else:
			classes[label] = classes[label]+1
			
	phrasefile.close()
	
vocabs = basevocab.keys()
vocabs.sort()
vocabindices = {}

for i in range(len(vocabs)):
	vocabindices[vocabs[i]] = i
	
labkeys = classes.keys()
labkeys.sort()
labindices = {}

for i in range(len(labkeys)):
	labindices[labkeys[i]] = i
	
labstrs = []

testbows = []
testlabstr = []
testlab = []
trainvbows = []
trainvlabstr = []
trainvlab = []
vvbows = []
vvlabstr = []
vvlab = []

totalwords = 0

with open(TEST_PHRASE_FILE_NAME,'r') as testphrases:
	for line in testphrases:
		sides = line.split('-')
		sides = [s.strip() for s in sides]
		testbows.append(makebow(vocabindices,[stems[s] for s in sides[0].split(' ')]))
		totalwords = totalwords + len(sides[0].split(' '))
		testlab.append(labindices[sides[1]])
		testlabstr.append(sides[1])
		
	testphrases.close()
	
if VALIDATION:	
	with open(VAL_PHRASE_FILE_NAME,'r') as valphrases:
		for line in valphrases:
			sides = line.split('-')
			sides = [s.strip() for s in sides]
			vvbows.append(makebow(vocabindices,[stems[s] for s in sides[0].split(' ')]))
			totalwords = totalwords + len(sides[0].split(' '))
			vvlab.append(labindices[sides[1]])
			vvlabstr.append(sides[1])
			
		valphrases.close()
		
	with open(TRAIN_PHRASE_FILE_NAME,'r') as trainphrases:
		for line in trainphrases:
			sides = line.split('-')
			sides = [s.strip() for s in sides]
			trainvbows.append(makebow(vocabindices,[stems[s] for s in sides[0].split(' ')]))
			totalwords = totalwords + len(sides[0].split(' '))
			trainvlab.append(labindices[sides[1]])
			trainvlabstr.append(sides[1])
			
		trainphrases.close()
else:
	with open(TRAIN_VAL_PHRASE_FILE_NAME,'r') as trainphrases:
		for line in trainphrases:
			sides = line.split('-')
			sides = [s.strip() for s in sides]
			trainvbows.append(makebow(vocabindices,[stems[s] for s in sides[0].split(' ')]))
			totalwords = totalwords + len(sides[0].split(' '))
			trainvlab.append(labindices[sides[1]])
			trainvlabstr.append(sides[1])
			
		trainphrases.close()

corrects = 0

klow = 3
kup = 3

if not VALIDATION:
	klow = kup

bestaccv = 0
bestk = klow-1
bestmodel = None

print "kNN"
print "Hyperparameter search..."

cond = False

if VALIDATION:
	cond = os.path.isfile('knn.params') and os.path.isfile('knn.model') and os.path.isfile('knn_val.acc')
else:
	cond = os.path.isfile('knn.model')

if cond:
	if VALIDATION:
		with open('knn_val.acc','r') as knnvalacc:
			for line in knnvalacc:
				bestaccv = float(line)
			knnvalacc.close()
		
		with open('knn.params','r') as knnparams:
			for line in knnparams:
				bestk = int(line)
			knnparams.close()
		
	with open('knn.model','rb') as bstmdl:
		bestmodel = cPickle.load(bstmdl)
		bstmdl.close()
else:
	for k in range(klow,kup+1):
		knncls = KNeighborsClassifier(n_neighbors=k,algorithm='kd_tree')
		knncls.fit(trainvbows,trainvlab)
		
		if VALIDATION:
			knnresults = knncls.predict(vvbows)
			cmatv = constructConfusionMatrix(len(classes.keys()))
			
			for j in range(len(vvbows)):
				cmatv[vvlab[j]][knnresults[j]] = cmatv[vvlab[j]][knnresults[j]]+1
				
			onevsv = oneVsAll(cmatv)
			onevsvacc = oneVsAllAcc(onevsv)
					
			print "Validation one-vs-all accuracy (k = %d) = %.2f%%" % (k,onevsvacc)
			
			if bestaccv < onevsvacc:
				bestmodel = knncls
				bestaccv = onevsvacc
				bestk = k
		else:
			bestmodel = knncls
		
	if VALIDATION:	
		with open('knn.params','w') as knnparams:
			knnparams.write('%d\n' % bestk)
			knnparams.close()
			
		with open('knn_val.acc','w') as knnvalacc:
			knnvalacc.write('%f\n' % bestaccv)
			knnvalacc.close()
		
	try:
		with open('knn.model','wb') as bstmdl:
			cPickle.dump(bestmodel,bstmdl)
			bstmdl.close()
	except MemoryError:
		print "Your model could not be saved."
	
print ""

if VALIDATION:
	print "Best k parameter has been chosen as %d (val. one-vs-all acc. = %.2f%%)" % (bestk,bestaccv)

cmat = None
testtime = 0

if os.path.isfile('knn_confusion.matrix') and os.path.isfile('knn_prediction.time'):
	cmat = loadCmatFromFile('knn_confusion.matrix')
	
	with open('knn_prediction.time','r') as knntime:
		for line in knntime:
			testtime = float(line)
		knntime.close()
else:
	start_time = time.time()
	knntest = bestmodel.predict(testbows)
	end_time = time.time()
	
	testtime = end_time - start_time
	
	cmat = constructConfusionMatrix(len(classes.keys()))

	for i in range(len(testbows)):
		cmat[testlab[i]][knntest[i]] = cmat[testlab[i]][knntest[i]]+1
		
	saveCmatToFile('knn_confusion.matrix',cmat)
	
	with open('knn_prediction.time','w') as knntime:
		knntime.write('%f\n' % testtime)
		knntime.close()
		
print "# of test instances = %d" % len(testbows) 
print "Avg. classification time of an instance = %.2f seconds" % (testtime/len(testbows))

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
