# python script to test the format of your submission

# your prediction results for each image
fileWriteSubmit = 'sampleSubmission.csv'

# the randomly generated ground-truth labels for each image in imgList.csv
fileGT = 'truth_random.csv'


with open(fileGT,'r') as f:
    lines = f.readlines()
truthVector = []
for line in lines:
    items = line.split()
    truthVector.append(int(items[0]))

with open(fileWriteSubmit,'r') as f:
    lines = f.readlines() 
predictionVector = []
predictionVector_top5 = []
for line in lines:
    items = line.rstrip().split(',')
    predictionVector.append(int(items[0]))
    if len(items)==5:
        predictionVector_top5.append([int(item) for item in items])
         
# compute confusion matrix 
n_classes = 205
confusionMat = [[0] * n_classes for i in range(n_classes)]
for pred, exp in zip(predictionVector, truthVector):
    confusionMat[pred][exp] += 1
t = sum(sum(l) for l in confusionMat)
# compute accuracy
accuracy = sum(confusionMat[i][i] for i in range(len(confusionMat)))*1.0 / t

# compute top 5 error
top5error = 'NA'
if len(predictionVector_top5) == len(truthVector):
    top5error = 0
    for i, curPredict in enumerate(predictionVector_top5):
        curTruth = truthVector[i]
        curHit = [1 for label in curPredict if label==curTruth]
        if len(curHit)==0:
            top5error = top5error+1
    top5error = top5error*1.0/len(truthVector)
            

print "accuracy:" + str(accuracy)
print "top 5 error rate:" + str(top5error)


#print "confusion matrix:"
#for line in confusionMat:
#    print ' '.join([str(item) for item in line]) + '\n'
