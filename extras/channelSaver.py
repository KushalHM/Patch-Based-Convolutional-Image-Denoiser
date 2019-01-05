import cv2
import numpy as np

file_path = './data/test'

saveDir = 'channelDir'
for path, subdirs, files in os.walk(file_path):
	for fName in files:
		print ("\n\n\n\nProcessing: ",fName)
		inFilePath = os.path.join(path, fName)
		print (inFilePath)
		filename = inFilePath.split('\\')[-1].split('.')[0]
		if not os.path.exists(saveDir):
			os.makedirs(saveDir)

		img = cv2.imread(inFilePath)
		
		outPathR = os.path.join(saveDir, filename+'_Red.jpg')
		outPathG = os.path.join(saveDir, filename+'_Green.jpg')
		outPathB = os.path.join(saveDir, filename+'_Blue.jpg')
		
		redPatch = patch[:,:,0]
		greenPatch = patch[:,:,1]
		bluePatch = patch[:,:,2]
		
		cv2.imwrite(outPathR,redPatch)
		cv2.imwrite(outPathG,greenPatch)
		cv2.imwrite(outPathB,bluePatch)
		
		print ("Images saved to: ", saveDir)
		
print ("Thank you for your patience. :)")