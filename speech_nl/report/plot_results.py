from confmatrix import *
import matplotlib.pyplot as plt
from matplotlib import cm

matrix_files = ['knn_confusion.matrix','../libshorttext-1.1/libshort_confusion_lr_unigram.matrix','../libshorttext-1.1/libshort_confusion_svm_unigram.matrix','../libshorttext-1.1/libshort_confusion_lr_bigram.matrix','../libshorttext-1.1/libshort_confusion_svm_bigram.matrix','../fastText/fasttext_confusion.matrix','../fastText/fasttext_confusion_bigram.matrix','deep_cnn_confusion.matrix','deep_rnn_confusion.matrix','deep_charbased_confusion.matrix']
arnames = ['stemming + BoW + kNN','unigram + LR','unigram + SVM','bigram + LR','bigram + SVM','unigram + Fast Text','bigram + Fast Text','word2vec + CNN','word2vec + RNN','CHARS-CNN']

distinct_classes = {}

PLOTS_WIDTH = 20
PLOTS_HEIGHT = 12

with open('analysis_corpus/testphrases_our.txt','r') as tphr:
	for line in tphr:
		sides = line.split('-')
		sides[1] = sides[1].strip()
		
		if not sides[1] in distinct_classes:
			distinct_classes[sides[1]] = True
			
	tphr.close()
	
class_sorted = distinct_classes.keys()
class_sorted.sort()

prs = []
rcs = []
fs = []

print "------One-vs-all accuracies------"
oneaccs = []

for i in range(len(arnames)):
	cmat = loadCmatFromFile(matrix_files[i])
	onevsall = oneVsAll(cmat)
	onevsallacc = oneVsAllAcc(onevsall)
	oneaccs.append(onevsallacc)
	prs.append(precision(cmat))
	rcs.append(recall(cmat))
	fs.append(fm(prs[len(prs)-1],rcs[len(rcs)-1]))
	
sortedindices = sorted(range(len(arnames)),reverse=True,key=lambda x: oneaccs[x])

for si in sortedindices:
	print "%s = %.2f%%" % (arnames[si],oneaccs[si])
	
fig = plt.figure(figsize=(PLOTS_WIDTH,PLOTS_HEIGHT))
plt.title('Precision rates of different architectures')
plt.xticks(range(len(class_sorted)),class_sorted,rotation='vertical')

for i in range(len(prs)):
	plt.plot(range(len(class_sorted)),prs[i],'-o',label=arnames[i])
	
plt.legend(loc='best')
plt.subplots_adjust(bottom=0.2)
fig.savefig('precisions.png')

fig = plt.figure(figsize=(PLOTS_WIDTH,PLOTS_HEIGHT))
plt.title('Recall rates of different architectures')
plt.xticks(range(len(class_sorted)),class_sorted,rotation='vertical')

for i in range(len(prs)):
	plt.plot(range(len(class_sorted)),rcs[i],'-o',label=arnames[i])
	
plt.legend(loc='best')
plt.subplots_adjust(bottom=0.2)
fig.savefig('recalls.png')

fig = plt.figure(figsize=(PLOTS_WIDTH,PLOTS_HEIGHT))
plt.title('F-scores of different architectures')
plt.xticks(range(len(class_sorted)),class_sorted,rotation='vertical')

for i in range(len(prs)):
	plt.plot(range(len(class_sorted)),fs[i],'-o',label=arnames[i])
	
plt.legend(loc='best')
plt.subplots_adjust(bottom=0.2)
fig.savefig('fscores.png')

print "Figures have been saved."
