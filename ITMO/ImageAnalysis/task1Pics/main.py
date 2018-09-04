#Ruben Cuadra
from scipy.misc import imread
from glob import glob
import numpy as np

#Folder contains many images in black&white in .bmp format
#we return a dict with {filename:imageMatrix}
def getDataFromFolder(pathToFolder):
	#==0 puts a True when color is black,False when white
	#then we cast that to Float, meaning 1.0 or 0.0
	return { p[len(pathToFolder)+1:]:(imread(p,mode="L")==0).astype(float) for p in glob(f"{pathToFolder}/*.bmp")}

#q is the added force
def runPotentialsAtReceptor(matrix,q=0.5):
	potentials = np.argwhere(matrix==1.0)
	#A lot of unecessary comparissons, maybe better do all in the center and handle borders differently
	for c,r in potentials: #Adjacent logic, need to verify that they exist
		if c+1 < matrix.shape[0]: 
			matrix[c+1][r] += q
			if r+1 < matrix.shape[1]: 
				matrix[c+1][r+1] += q #Diagonal +1,+1
		if c-1 >= 0: 			  
			matrix[c-1][r] += q
			if r-1 >= 0: 			  
				matrix[c-1][r-1] += q #Diagonal -1,-1
		if r+1 < matrix.shape[1]: 
			matrix[c][r+1] += q
			if c-1 >= 0:   
				matrix[c-1][r+1] += q #Diagonal -1,+1
		if r-1 >= 0: 			  
			matrix[c][r-1] += q
			if c+1 < matrix.shape[0]:
				matrix[c+1][r-1] += q #Diagonal +1,-1
	return matrix

def predict(imgToPredict, processedImages):
	toPredPotentials = np.argwhere(imgToPredict==1.0) #get row,column where image is black
	imageName,bestVal = "error", float("-inf") #Init returns
	for fileP,imgMatrix in processedImages.items():
		cVal = 0	#Reset counter
		for c,r in toPredPotentials: cVal+=imgMatrix[c][r] #Sum vals with processed matrix
		if cVal > bestVal: #Update best val
			bestVal = cVal
			imageName = fileP
	return imageName #To the corresponding classification


if __name__ == '__main__':
	#Init Paths
	trainPath = "train"
	testPath  = "test"

	#Processing
	print(f'Processing images from folder "{trainPath}/"')
	trainData,processed = getDataFromFolder(trainPath),{}
	for fileP,imgMatrix in trainData.items():
		processed[fileP] = runPotentialsAtReceptor(imgMatrix,q=0.1)
		print(f'\t{chr(10003)} {fileP}')

	#Predict
	print(f'Predicting images from folder "{testPath}/"')
	testData = getDataFromFolder(testPath)
	for fileP,imgMatrix in testData.items():
		result = predict(imgMatrix,processed)
		print(f'\t{fileP} = {result}')
