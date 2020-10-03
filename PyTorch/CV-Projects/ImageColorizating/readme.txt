This is the test set for Places-205 database. There are 41,000 images (200 images per category. We only release the test images without any ground-truth, you could submit your prediction result to our evaluation server (http://places.csail.mit.edu) to see how your trained model's performance on this test set.

The prediction result should be in the format of the sample result file sampleSubmission.csv: Each line of prediction are aligned with the image name in imgList.csv. And each line contains 5 possible prediction labels of your algorithm, splited by ','. The 1st label  will be used to compute the accuracy averaged over categories, and all the 5 labels will be used to compute the Top 5 error rate. Note that the category index should be between [0, 204], rather than [1,205].

You must make sure your submission is in the correct format, or our evaluation server will fail to recognize it. You could first use testSampleScript.py to test your submission. The evaluaiton script on our server side is pretty much similar to that script. 

Bolei Zhou
Oct.24, 2014
