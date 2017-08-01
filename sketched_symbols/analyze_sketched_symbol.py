from sketchfe.FeatureExtractor import *
from os import listdir
from os.path import isfile,join
from sketchfe import shapecreator
from random import shuffle
from math import *
import matplotlib.pyplot as plt
from matplotlib import cm
import locale
import sys
sys.path.append('./python')

from svmutil import *

locale.setlocale(locale.LC_NUMERIC,'C')

generic_path = '../soccer_annotated_sketches/'

classes=['player_motion','ball_motion','player_position1','player_position2']
class_counts = {}
class_correct_counts = {}

for cl in classes:
	class_counts[cl] = 0
	class_correct_counts[cl] = 0

files_plmovement = [join(generic_path+'playermotion/sketch/',f) for f in listdir(generic_path+'playermotion/sketch/') if isfile(join(generic_path+'playermotion/sketch/',f))]
files_ballmovement = [join(generic_path+'ballmotion/sketch/',f) for f in listdir(generic_path+'ballmotion/sketch/') if isfile(join(generic_path+'ballmotion/sketch/',f))]
files_plposition = [join(generic_path+'player/sketch/',f) for f in listdir(generic_path+'player/sketch/') if isfile(join(generic_path+'player/sketch/',f))]
files_otherplposition = [join(generic_path+'player_opposite/sketch/',f) for f in listdir(generic_path+'player_opposite/sketch/') if isfile(join(generic_path+'player_opposite/sketch/',f))]

labels = []
feats = []

labelcounts = [0 for i in range(4)]

for plm in files_plmovement:
	with open(plm,'rb') as file_plm:
		print "Extracting %s" % plm
		loadedSketch = shapecreator.buildSketch('school',file_plm.read().replace(',','.'))
		fextractor = IDMFeatureExtractor()
		features = fextractor.extract(loadedSketch).tolist()
		feats.append(features)
		labels.append(1)
		labelcounts[0] = labelcounts[0]+1
		
for plm in files_ballmovement:
	with open(plm,'rb') as file_plm:
		print "Extracting %s" % plm
		loadedSketch = shapecreator.buildSketch('school',file_plm.read().replace(',','.'))
		fextractor = IDMFeatureExtractor()
		features = fextractor.extract(loadedSketch).tolist()
		feats.append(features)
		labels.append(2)
		labelcounts[1] = labelcounts[1]+1
		
for plm in files_plposition:
	with open(plm,'rb') as file_plm:
		print "Extracting %s" % plm
		loadedSketch = shapecreator.buildSketch('school',file_plm.read().replace(',','.'))
		fextractor = IDMFeatureExtractor()
		features = fextractor.extract(loadedSketch).tolist()
		feats.append(features)
		labels.append(3)
		labelcounts[2] = labelcounts[2]+1
		
for plm in files_otherplposition:
	with open(plm,'rb') as file_plm:
		print "Extracting %s" % plm
		loadedSketch = shapecreator.buildSketch('school',file_plm.read().replace(',','.'))
		fextractor = IDMFeatureExtractor()
		features = fextractor.extract(loadedSketch).tolist()
		feats.append(features)
		labels.append(4)
		labelcounts[3] = labelcounts[3]+1
		
limits_per_class = [4*i/5 for i in labelcounts]
limits_per_class_tv = [4*i/5 for i in limits_per_class]
curlabcounts = [0 for i in range(4)]
		
num_instances = len(labels)
indiceslist = range(0,num_instances)
shuffle(indiceslist)

acccum = 0

trset = []
trlab = []
vset = []
vlab = []
testset = []
testlab = []
	
for i in range(0,num_instances):
	classno = labels[indiceslist[i]]-1
	
	if curlabcounts[classno] <= limits_per_class_tv[classno]:
		trset.append(feats[indiceslist[i]])
		trlab.append(labels[indiceslist[i]])
	elif curlabcounts[classno] <= limits_per_class[classno]:
		vset.append(feats[indiceslist[i]])
		vlab.append(labels[indiceslist[i]])
	else:
		testset.append(feats[indiceslist[i]])
		testlab.append(labels[indiceslist[i]])
		
	curlabcounts[classno] = curlabcounts[classno]+1
	
print "Set sizes: "
print "Training = %d" % len(trset)
print "Validation = %d" % len(vset)
print "Testing = %d" % len(testset)
		
prob = svm_problem(trlab,trset,isKernel=True)

coststart = -1
costend = 10
gammastart = -10
gammaend = 1

bestcost = coststart
bestgamma = gammastart
bestacc = 0
overacc = 0
absolutebestmodel = None

while coststart <= costend:
	gammastart = -10
	
	while gammastart <= gammaend:
		param = svm_parameter('-t 3 -c %f -g %f -q' % (2**coststart,2**gammastart))
		model = svm_train(prob,param)
		(l,acc,dv) = svm_predict(vlab,vset,model)
		
		if bestacc < acc[0]:
			bestacc = acc[0]
			bestcost = coststart
			bestgamma = gammastart
			absolutebestmodel = model
		
		gammastart  = gammastart + 1
		
	coststart = coststart + 1
	
print "bestcost in log2 = %d, bestgamma in log2 = %d, bestacc in perc. = %f" % (bestcost,bestgamma,bestacc)

(l,acc,dv) = svm_predict(testlab,testset,absolutebestmodel)

for insno,corlab in enumerate(l):
	class_counts[classes[testlab[insno]-1]] = class_counts[classes[testlab[insno]-1]]+1
	
	if testlab[insno] == l[insno]:
		class_correct_counts[classes[testlab[insno]-1]] = class_correct_counts[classes[testlab[insno]-1]]+1

plt.title('IDM+SVM classification results for each class (overall %.2f%%)' % (acc[0]))
plt.xticks(range(len(classes)),classes)
y = np.array([100.0*class_correct_counts[cl]/class_counts[cl] for cl in classes])
plt.bar(range(len(classes)),y,facecolor="#9999ff",edgecolor="white",align="center")

ind = 0
for cl in classes:
	plt.text(ind,y[ind]+0.05,'%d/%d (%.2f%%)' % (class_correct_counts[cl],class_counts[cl],100.0*class_correct_counts[cl]/class_counts[cl]),ha="center",va="bottom")
	ind = ind+1

plt.show()

svm_save_model('symbol_recognizer.model',absolutebestmodel)

'''
plt.title('# of symbols for each class')
print classes
cs=cm.Set1(np.arange(40)/40.)
plt.pie([class_counts[cl] for cl in classes],explode=[0]*len(classes),labels=['%s (%d)' % (cl,class_counts[cl]) for cl in classes],autopct='%.2f%%',shadow=False,startangle=90,colors=cs)
plt.show()'''

