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
