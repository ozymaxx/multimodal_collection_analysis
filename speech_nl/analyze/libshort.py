from subprocess import Popen,PIPE,call
from libshorttext.analyzer import *
import os
import time 
import sys

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
sys.path.append('../soccerretrieval_semantic')

from confmatrix import *

VALIDATION = False
		
def parseConfusionMatrix(orig_file,our_labindices):
	cmat = constructConfusionMatrix(len(our_labindices.keys()))
	unsorted_labs = None
	cmat_orig = []
	
	ind = 0
	with open(orig_file,'r') as origfile:
		for line in origfile:
			if ind == 0:
				unsorted_labs = line.split()
			else:
				linsplit = line.split()
				linsplit = linsplit[1:len(linsplit)]
				cmat_orig.append([int(nm) for nm in linsplit])
				
			ind = ind+1
		origfile.close()
		
	#print unsorted_labs
	#print cmat_orig
		
	orig_labindices = {}
	for i in range(len(unsorted_labs)):
		orig_labindices[unsorted_labs[i]] = i
		
	for label1 in unsorted_labs:
		for label2 in unsorted_labs:
			cmat[our_labindices[label1]][our_labindices[label2]] = cmat_orig[orig_labindices[label1]][orig_labindices[label2]]
			
	return cmat

distinct_classes = {}

with open('testphrases_libshort.txt','r') as testfile:
	for line in testfile:
		sides = line.split('\t')
		
		if not sides[0] in distinct_classes:
			distinct_classes[sides[0]] = True
			
	testfile.close()
	
classes_sorted = distinct_classes.keys()
classes_sorted.sort()

our_labindices = {}

for i in range(len(classes_sorted)):
	our_labindices[classes_sorted[i]] = i
	
cond = False

if VALIDATION:
	cond = not os.path.isfile('trainphrases_libshort_svm_bigram.txt.model')
else:
	cond = not os.path.isfile('trainvalphrases_libshort_svm_bigram.txt.model')

if cond:
	if VALIDATION:
		call(['python text-train.py trainphrases_libshort_svm_bigram.txt'],shell=True)
	else:
		call(['python text-train.py trainvalphrases_libshort_svm_bigram.txt'],shell=True)
	
if not os.path.isfile('libshort_results_svm_bigram'):
	if VALIDATION:
		call(['python text-predict.py testphrases_libshort.txt trainphrases_libshort_svm_bigram.txt.model libshort_results_svm_bigram'],shell=True)
	else:
		call(['python text-predict.py testphrases_libshort.txt trainvalphrases_libshort_svm_bigram.txt.model libshort_results_svm_bigram'],shell=True)
	
cmat = None

if not os.path.isfile('libshort_confusion_svm_bigram.matrix'):
	results = InstanceSet('libshort_results_svm_bigram')
	
	analyzer = None
	if VALIDATION:
		analyzer = Analyzer('trainphrases_libshort_svm_bigram.txt.model')
	else:
		analyzer = Analyzer('trainvalphrases_libshort_svm_bigram.txt.model')
		
	analyzer.gen_confusion_table(results,output='libshort_conf_svm_bigram.matrix')
	cmat = parseConfusionMatrix('libshort_conf_svm_bigram.matrix',our_labindices)
	saveCmatToFile('libshort_confusion_svm_bigram.matrix',cmat)
else:
	cmat = loadCmatFromFile('libshort_confusion_svm_bigram.matrix')
	
onevsall = oneVsAll(cmat)

print "One-vs-all accuracy = %.2f%%" % oneVsAllAcc(onevsall)
print ""

print "Precision rates for each class"
precisions = precision(cmat)

for i in range(len(classes_sorted)):
	print "%s = %.2f%%" % (classes_sorted[i],precisions[i])
	
print "Recall rates for each class"
recalls = recall(cmat)

for i in range(len(classes_sorted)):
	print "%s = %.2f%%" % (classes_sorted[i],recalls[i])
	
print "F1 rates for each class"
fs = fm(precisions,recalls)

for i in range(len(classes_sorted)):
	print "%s = %.2f%%" % (classes_sorted[i],fs[i])
	
	
print ""
print ""

cond = False

if VALIDATION:
	cond = not os.path.isfile('trainphrases_libshort_lr_bigram.txt.model')
else:
	cond = not os.path.isfile('trainvalphrases_libshort_lr_bigram.txt.model')

if cond:
	if VALIDATION:
		call(['python text-train.py -L 3 trainphrases_libshort_lr_bigram.txt'],shell=True)
	else:
		call(['python text-train.py -L 3 trainvalphrases_libshort_lr_bigram.txt'],shell=True)
	
if not os.path.isfile('libshort_results_lr_bigram'):
	if VALIDATION:
		call(['python text-predict.py testphrases_libshort.txt trainphrases_libshort_lr_bigram.txt.model libshort_results_lr_bigram'],shell=True)
	else:
		call(['python text-predict.py testphrases_libshort.txt trainvalphrases_libshort_lr_bigram.txt.model libshort_results_lr_bigram'],shell=True)
	
cmat = None

if not os.path.isfile('libshort_confusion_lr_bigram.matrix'):
	results = InstanceSet('libshort_results_lr_bigram')
	
	analyzer = None
	if VALIDATION:
		analyzer = Analyzer('trainphrases_libshort_lr_bigram.txt.model')
	else:
		analyzer = Analyzer('trainvalphrases_libshort_lr_bigram.txt.model')
		
	analyzer.gen_confusion_table(results,output='libshort_conf_lr_bigram.matrix')
	cmat = parseConfusionMatrix('libshort_conf_lr_bigram.matrix',our_labindices)
	saveCmatToFile('libshort_confusion_lr_bigram.matrix',cmat)
else:
	cmat = loadCmatFromFile('libshort_confusion_lr_bigram.matrix')
	
onevsall = oneVsAll(cmat)

print "One-vs-all accuracy = %.2f%%" % oneVsAllAcc(onevsall)
print ""

print "Precision rates for each class"
precisions = precision(cmat)

for i in range(len(classes_sorted)):
	print "%s = %.2f%%" % (classes_sorted[i],precisions[i])
	
print "Recall rates for each class"
recalls = recall(cmat)

for i in range(len(classes_sorted)):
	print "%s = %.2f%%" % (classes_sorted[i],recalls[i])
	
print "F1 rates for each class"
fs = fm(precisions,recalls)

for i in range(len(classes_sorted)):
	print "%s = %.2f%%" % (classes_sorted[i],fs[i])
	
print ""
print ""

cond = False

if VALIDATION:
	cond = not os.path.isfile('trainphrases_libshort_svm_unigram.txt.model')
else:
	cond = not os.path.isfile('trainvalphrases_libshort_svm_unigram.txt.model')

if cond:
	if VALIDATION:
		call(['python text-train.py -P 0 trainphrases_libshort_svm_unigram.txt'],shell=True)
	else:
		call(['python text-train.py -P 0 trainvalphrases_libshort_svm_unigram.txt'],shell=True)
	
if not os.path.isfile('libshort_results_svm_unigram'):
	if VALIDATION:
		call(['python text-predict.py testphrases_libshort.txt trainphrases_libshort_svm_unigram.txt.model libshort_results_svm_unigram'],shell=True)
	else:
		call(['python text-predict.py testphrases_libshort.txt trainvalphrases_libshort_svm_unigram.txt.model libshort_results_svm_unigram'],shell=True)
	
cmat = None

if not os.path.isfile('libshort_confusion_svm_unigram.matrix'):
	results = InstanceSet('libshort_results_svm_unigram')
	
	analyzer = None
	
	if VALIDATION:
		analyzer = Analyzer('trainphrases_libshort_svm_unigram.txt.model')
	else:
		analyzer = Analyzer('trainvalphrases_libshort_svm_unigram.txt.model')
		
	analyzer.gen_confusion_table(results,output='libshort_conf_svm_unigram.matrix')
	cmat = parseConfusionMatrix('libshort_conf_svm_unigram.matrix',our_labindices)
	saveCmatToFile('libshort_confusion_svm_unigram.matrix',cmat)
else:
	cmat = loadCmatFromFile('libshort_confusion_svm_unigram.matrix')
	
onevsall = oneVsAll(cmat)

print "One-vs-all accuracy = %.2f%%" % oneVsAllAcc(onevsall)
print ""

print "Precision rates for each class"
precisions = precision(cmat)

for i in range(len(classes_sorted)):
	print "%s = %.2f%%" % (classes_sorted[i],precisions[i])
	
print "Recall rates for each class"
recalls = recall(cmat)

for i in range(len(classes_sorted)):
	print "%s = %.2f%%" % (classes_sorted[i],recalls[i])
	
print "F1 rates for each class"
fs = fm(precisions,recalls)

for i in range(len(classes_sorted)):
	print "%s = %.2f%%" % (classes_sorted[i],fs[i])
	
	
print ""
print ""

cond = False

if VALIDATION:
	cond = not os.path.isfile('trainphrases_libshort_lr_unigram.txt.model')
else:
	cond = not os.path.isfile('trainvalphrases_libshort_lr_unigram.txt.model')

if cond:
	if VALIDATION:
		call(['python text-train.py -P 0 -L 3 trainphrases_libshort_lr_unigram.txt'],shell=True)
	else:
		call(['python text-train.py -P 0 -L 3 trainvalphrases_libshort_lr_unigram.txt'],shell=True)
	
if not os.path.isfile('libshort_results_lr_unigram'):
	if VALIDATION:
		call(['python text-predict.py testphrases_libshort.txt trainphrases_libshort_lr_unigram.txt.model libshort_results_lr_unigram'],shell=True)
	else:
		call(['python text-predict.py testphrases_libshort.txt trainvalphrases_libshort_lr_unigram.txt.model libshort_results_lr_unigram'],shell=True)
	
cmat = None

if not os.path.isfile('libshort_confusion_lr_unigram.matrix'):
	results = InstanceSet('libshort_results_lr_unigram')
	
	analyzer = None
	if VALIDATION:
		analyzer = Analyzer('trainphrases_libshort_lr_unigram.txt.model')
	else:
		analyzer = Analyzer('trainvalphrases_libshort_lr_unigram.txt.model')
		
	analyzer.gen_confusion_table(results,output='libshort_conf_lr_unigram.matrix')
	cmat = parseConfusionMatrix('libshort_conf_lr_unigram.matrix',our_labindices)
	saveCmatToFile('libshort_confusion_lr_unigram.matrix',cmat)
else:
	cmat = loadCmatFromFile('libshort_confusion_lr_unigram.matrix')
	
onevsall = oneVsAll(cmat)

print "One-vs-all accuracy = %.2f%%" % oneVsAllAcc(onevsall)
print ""

print "Precision rates for each class"
precisions = precision(cmat)

for i in range(len(classes_sorted)):
	print "%s = %.2f%%" % (classes_sorted[i],precisions[i])
	
print "Recall rates for each class"
recalls = recall(cmat)

for i in range(len(classes_sorted)):
	print "%s = %.2f%%" % (classes_sorted[i],recalls[i])
	
print "F1 rates for each class"
fs = fm(precisions,recalls)

for i in range(len(classes_sorted)):
	print "%s = %.2f%%" % (classes_sorted[i],fs[i])
