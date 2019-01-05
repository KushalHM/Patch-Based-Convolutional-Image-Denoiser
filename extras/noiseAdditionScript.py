import cv2
from utils import *
import os
from datetime import datetime
import csv

startTime = datetime.now()
currDir = os.getcwd()
dataPath = os.path.join(currDir, "2")
outPath = dataPath+'_noised'
dataFileName = 'dataFile.csv'
csvData = []

if not os.path.exists(outPath):
    os.makedirs(outPath)

i=0
failed = [];

print('Image read location: ',dataPath)
print('Image write locaton: ',outPath)

with open(dataFileName,'w') as file:
    file.write('originalImageLoc,noisedImageLoc\n')
for path, subdirs, files in os.walk(dataPath):
    for fName in files:
        dataLine = []
        inFilePath = os.path.join(path, fName)
        outFilePath = os.path.join(outPath,os.path.basename(path)+'_'+fName)
        img = cv2.imread(inFilePath)
        dataLine.append(inFilePath)
        try:
            imgNoised = addMultiLayerGaussian(img)
            cv2.imwrite(outFilePath,imgNoised)
            dataLine.append(outFilePath)
            i+=1
            csvData.append(dataLine)
            if(i%1000 == 0):
                with open(dataFileName,'a') as file:
                    writer = csv.writer(file)
                    writer.writerows(csvData)
                print("Number of images processed: ",i)
                csvData = []
        except:
            print ("Failed for file: ",outFilePath)
            failed.append(outFilePath+'\n')

with open(dataFileName,'a') as file:
    writer = csv.writer(file)
    writer.writerows(csvData)
print (csvData)
print("Total time taken: ", datetime.now() - startTime)
print("Total files processed: ",i)
print("Failed file names: ",failed)
with open('failedFiles.txt','a') as file:
    file.writelines(failed)
