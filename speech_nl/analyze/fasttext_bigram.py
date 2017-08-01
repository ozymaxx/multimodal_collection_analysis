from subprocess import Popen,PIPE,call
import os
import time 

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import sys
sys.path.append('../soccerretrieval_semantic')

from confmatrix import *

VALIDATION = False

if not os.path.isfile('model_fb_bigram.bin'):
	if VALIDATION:
		call(['./fasttext supervised -input trainphrases_fb.txt -output model_fb_bigram -wordNgrams 2'],shell=True)
	else:
		call(['./fasttext supervised -input trainvalphrases_fb.txt -output model_fb_bigram -wordNgrams 2'],shell=True)

golds = []
corrects = 0
alls = 0
class_dist = {}
class_dist_preds = {}

with open('testphrases_fb.txt','r') as testp:
	for line in testp:
		if len(line) > 0:
			tokens = line.split(' ')
			label = tokens[0][9:len(tokens[0])]
			golds.append(label)
			
			if not label in class_dist:
				class_dist[label] = 1
				class_dist_preds[label] = 0
			else:
				class_dist[label] = class_dist[label]+1
				
			alls = alls+1
		
	testp.close()
	
class_to_ix = {}
classsorted = class_dist.keys()
classsorted.sort()

for i in range(len(classsorted)):
	class_to_ix[classsorted[i]] = i
	
testtime = 0
cmat = None
	
if os.path.isfile('fasttext_confusion_bigram.matrix') and os.path.isfile('fasttext_prediction_bigram.time'):
	with open('fasttext_prediction_bigram.time','r') as ftime:
		for line in ftime:
			testtime = float(line)
		ftime.close()
		
	cmat = loadCmatFromFile('fasttext_confusion_bigram.matrix')
else:
	start_time = time.time()

	p = Popen(['./fasttext predict model_fb_bigram.bin testphrases_fb.txt 1'],shell=True,stdin=PIPE,stdout=PIPE,stderr=PIPE)
	output,err = p.communicate()

	end_time = time.time()
	
	testtime = end_time - start_time

	classlines = output.split('\n')
	
	cmat = constructConfusionMatrix(len(classsorted))

	for i in range(len(golds)):
		result = classlines[i]
		cmat[class_to_ix[golds[i]]][class_to_ix[result[9:len(result)]]] = cmat[class_to_ix[golds[i]]][class_to_ix[result[9:len(result)]]]+1
				
	saveCmatToFile('fasttext_confusion_bigram.matrix',cmat)
	
	with open('fasttext_prediction_bigram.time','w') as ftime:
		ftime.write('%f\n' % testtime)
		ftime.close()
		
print "# of test instances = %d" % len(golds) 
print "Avg. classification time of an instance = %.2f seconds" % (testtime/len(golds))

onevsall = oneVsAll(cmat)

print "One-vs-all accuracy = %.2f%%" % oneVsAllAcc(onevsall)
print ""

print "Precision rates for each class"
precisions = precision(cmat)

for i in range(len(classsorted)):
	print "%s = %.2f%%" % (classsorted[i],precisions[i])
	
print "Recall rates for each class"
recalls = recall(cmat)

for i in range(len(classsorted)):
	print "%s = %.2f%%" % (classsorted[i],recalls[i])
	
print "F1 rates for each class"
fs = fm(precisions,recalls)

for i in range(len(classsorted)):
	print "%s = %.2f%%" % (classsorted[i],fs[i])
