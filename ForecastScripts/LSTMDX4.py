from pandas import DataFrame
from pandas import read_csv
import random
import math
import numpy as np 
import tensorflow  as tf
from tensorflow.examples.tutorials.mnist import input_data
from numpy import zeros, newaxis
from sklearn.model_selection import train_test_split
# np.set_printoptions(threshold='nan')

def unison_shuffled_copies(a, b,c):
    assert len(a) == len(b)
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p] , c[p]


def unison_shuffled_copiesFull(a, b,c,d,e):
    assert len(a) == len(b)
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p] , c[p] , d[p],e[p]




def Map(input,step=19):
	if input==1:
		#return [[1,0,0] for i in range(step)]  
		return [1,0,0]
	elif input==2:
		#return [[0,1,0] for i in range(step)]  
		return [0,1,0]
	elif input==3:
		#return [[0,0,1] for i in range(step)]  
		return [0,0,1] 

def loss_function( target,outputs,seq_length , alpha):

	output = tf.nn.softmax(outputs)
	# Compute cross entropy for output frame.
	cross_entropy = target * tf.log(output)
	cross_entropy = -tf.reduce_sum(cross_entropy, 2)
	mask = tf.sign(tf.reduce_max(tf.abs(target), 2))
	cross_entropy *= mask
	# Average over actual sequence lengths.
	cross_entropy = tf.reduce_sum(cross_entropy, 1)
	cross_entropy /= tf.reduce_sum(mask, 1)
	lasty = last_relevant(target, seq_length)
	lastOutput = last_relevant(outputs, seq_length)
	loss = tf.nn.softmax_cross_entropy_with_logits(labels = lasty,logits = lastOutput)
	cross_entropy = cross_entropy*alpha + (1-2*alpha)*loss

	return tf.reduce_mean(cross_entropy)







def RemoveLastValue(DX):
	cols = DX.columns.values
	for i in range (len(cols)):
		if(cols[i]=="114" or cols[i].startswith( "114." )):
				DX.drop(cols[i], axis=1, inplace=True)
	val=19
	DX_values = DX.values
	indx=[]
	for i in range(DX_values.shape[0]):
		if(i==val):
			val+=20
		else:
			indx.append(i)
	DX_values=DX_values[indx]
	return DX_values

def last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant

def mergeFiles(inFileDX ,  inFileCDRSB , inFileMMSE, isTesting=False):

	dataDX = '/Users/Tim/Desktop/Alzheimer/ForecastProcessed_data/'+inFileDX+'.csv'
	dataCDRSB = '/Users/Tim/Desktop/Alzheimer/ForecastProcessed_data/'+inFileCDRSB+'.csv'
	dataMMSE = '/Users/Tim/Desktop/Alzheimer/ForecastProcessed_data/'+inFileMMSE+'.csv'

	dataframeDX = read_csv(dataDX, names=None)
	dataframeCDRSB = read_csv(dataCDRSB, names=None)
	dataframeMMSE = read_csv(dataMMSE, names=None)
	if(not isTesting):
		dataframeDX = RemoveLastValue(dataframeDX)
	else:
		dataframeDX = dataframeDX.values

	dataframeCDRSB = dataframeCDRSB.values
	dataframeMMSE = dataframeMMSE.values
	numberOfFeaturesCDRSB = int((dataframeCDRSB.shape[1]-2)/19)
	numberOfFeaturesMMSE = int((dataframeMMSE.shape[1]-2)/19)
	numberOfFeaturesDx =  int((dataframeDX.shape[1]-2)/19)
	# print numberOfFeaturesDx ,numberOfFeaturesCDRSB ,  numberOfFeaturesMMSE 
	# print dataframeDX.shape, dataframeCDRSB.shape , dataframeMMSE.shape
	numberOfFeatures = int(numberOfFeaturesCDRSB +  numberOfFeaturesMMSE+ numberOfFeaturesDx)

	if(not isTesting):

		input = np.zeros((dataframeMMSE.shape[0] ,numberOfFeatures*19+4))
		input[:,0] = dataframeDX[:,0]
		input[:,-3] = dataframeDX[:,-1]
		input[:,-2] = dataframeCDRSB[:,-1]
		input[:,-1] = dataframeMMSE[:,-1]
		for i in range(0, input.shape[0]):
			for step in range (19):
				for DxFeature in range(int(numberOfFeaturesDx)):
					# print "DX" ,  1+DxFeature+step*numberOfFeatures  , 1+step*numberOfFeaturesDx+DxFeature
					input[i,1+DxFeature+step*numberOfFeatures]=dataframeDX[i,1+step*numberOfFeaturesDx+DxFeature]
				# print "DX" ,  1+numberOfFeaturesDx+step*numberOfFeatures   , 1+step*numberOfFeaturesDx+DxFeature
				for CDRSBFeature in range(int(numberOfFeaturesCDRSB)):
					input[i,1+numberOfFeaturesDx+CDRSBFeature+step*numberOfFeatures ] = dataframeCDRSB[i,1+step*numberOfFeaturesCDRSB+CDRSBFeature]
				for MMSEFeature in range(int(numberOfFeaturesMMSE)):
					input[i,1+numberOfFeaturesDx+numberOfFeaturesCDRSB + MMSEFeature+step*numberOfFeatures ] = dataframeMMSE[i,1+step*numberOfFeaturesMMSE+MMSEFeature]
		cols=["RID"]
		for i in range(19):
			for DxFeature in range(int(numberOfFeaturesDx)):
				cols.append("DX_f_"+str(DxFeature)+"_"+str(i))
			for CDRSBFeature in range(int(numberOfFeaturesCDRSB)):
				cols.append("CDRSB_"+str(CDRSBFeature)+str(i+1))
			for MMSEFeature in range(int(numberOfFeaturesMMSE)):
				cols.append("MMSE_"+str(MMSEFeature)+str(i+1))
		cols.append("DX_Target")
		cols.append("CDRSB_Target")
		cols.append("MMSE_Target")
		# print "train col" ,  len(cols)
	else:
		input = np.zeros((dataframeMMSE.shape[0] ,numberOfFeatures*19+1))
		input[:,0] = dataframeDX[:,0]
		for i in range(0, input.shape[0]):
			for step in range (19):
				for DxFeature in range(int(numberOfFeaturesDx)):
					# print "DX" ,  1+DxFeature+step*numberOfFeatures  , 1+step*numberOfFeaturesDx+DxFeature
					input[i,1+DxFeature+step*numberOfFeatures]=dataframeDX[i,1+step*numberOfFeaturesDx+DxFeature]
				# print "DX" ,  1+numberOfFeaturesDx+step*numberOfFeatures   , 1+step*numberOfFeaturesDx+DxFeature
				for CDRSBFeature in range(int(numberOfFeaturesCDRSB)):
					input[i,1+numberOfFeaturesDx+CDRSBFeature+step*numberOfFeatures ] = dataframeCDRSB[i,1+step*numberOfFeaturesCDRSB+CDRSBFeature]
				for MMSEFeature in range(int(numberOfFeaturesMMSE)):
					input[i,1+numberOfFeaturesDx+numberOfFeaturesCDRSB + MMSEFeature+step*numberOfFeatures ] = dataframeMMSE[i,1+step*numberOfFeaturesMMSE+MMSEFeature]
		cols=["RID"]
		for i in range(19):
			for DxFeature in range(int(numberOfFeaturesDx)):
				cols.append("DX_f_"+str(DxFeature)+"_"+str(i))
			for CDRSBFeature in range(int(numberOfFeaturesCDRSB)):
				cols.append("CDRSB_"+str(CDRSBFeature)+str(i+1))
			for MMSEFeature in range(int(numberOfFeaturesMMSE)):
				cols.append("MMSE_"+str(MMSEFeature)+str(i+1))
		# print "test col" , len(cols)
	df = DataFrame(input , columns = cols)
	if(isTesting):
		df.to_csv('/Users/Tim/Desktop/Alzheimer/ForecastProcessed_data/LSTMParsedTestDataTempDX_andFeatures.csv',index=False)
	else:
		df.to_csv('/Users/Tim/Desktop/Alzheimer/ForecastProcessed_data/LSTMParsedTrainDataTempDX_andFeatures.csv',index=False)
	return df , numberOfFeaturesDx ,  numberOfFeaturesCDRSB  ,  numberOfFeaturesMMSE



def LSTMDX(TrainDX,TrainCDRSB, TrainMMSE , TestDX,TestCDRSB,TestMMSE , Validation=False):
	TrainDataframe , numberOfFeaturesDxTrain, numberOfFeaturesCDRSBTrain  ,  numberOfFeaturesMMSETrain    =mergeFiles(TrainDX , TrainCDRSB ,TrainMMSE)
	TestDataframe, numberOfFeaturesDxTest ,  numberOfFeaturesCDRSBTest  ,  numberOfFeaturesMMSETest   =mergeFiles(TestDX , TestCDRSB , TestMMSE ,isTesting=True)

	################################ Comment
	#TrainData = '/Users/Tim/Desktop/Alzheimer/ForecastProcessed_data/LSTMParsedTrainDataTempDX_andFeatures.csv'
	#TestData ='/Users/Tim/Desktop/Alzheimer/ForecastProcessed_data/LSTMParsedTestDataTempDX_andFeatures.csv'

	#TrainDataframe = read_csv(TrainData, names=None)
	#TestDataframe = read_csv(TestData, names=None)
	#numberOfFeaturesDxTest = 2
	#numberOfFeaturesCDRSBTest =6
	#numberOfFeaturesMMSETest = 7
	###########################################

	TrainInfo=TrainDataframe.values
	# print TrainInfo[-1,:]
	TestInfo=TestDataframe.values
	numberOfFeatures=(TrainInfo.shape[1]-2)/19
	#print numberOfFeatures , TrainDataframe.shape , TestDataframe.shape
	################### Training Data ##############################
	TrainOutput = TrainInfo[:,-3]
	TrainOutputCDRSB = TrainInfo[:,-2]
	TrainOutputMMSE = TrainInfo[:,-1]
	TrainData = TrainInfo[:,1:-3]
	Trainseq_length=[]
	for i in range(TrainData.shape[0]):
		seq=0
		for j in range (0,TrainData.shape[1],int(numberOfFeatures)):
			if(TrainData[i,j]!=0):
				seq+=1
		Trainseq_length.append(seq)
	Trainseq_length = np.array(Trainseq_length)

	CompleteTrainOutput = np.zeros((TrainOutput.shape[0],19,3))
	CompleteTrainOutputCDRSB = np.zeros((TrainOutput.shape[0],19,1))
	CompleteTrainOutputMMSE = np.zeros((TrainOutput.shape[0],19,1))
	# TrainOutputCDRSB=TrainOutputCDRSB.reshape(TrainOutputCDRSB.shape[0],19,1)
	# TrainOutputMMSE=TrainOutputMMSE.reshape(TrainOutputMMSE.shape[0],19,1)

	for i in range (TrainOutput.shape[0]):
		for j in range(Trainseq_length[i]):
			CompleteTrainOutput[i,j,:] =(Map(TrainOutput[i]))
			CompleteTrainOutputCDRSB [i,j,0]= TrainOutputCDRSB[i]
			CompleteTrainOutputMMSE [i,j,0]= TrainOutputMMSE[i]
	TrainData =  TrainData.reshape((int(TrainData.shape[0]), 19 ,int(numberOfFeatures) ))
		################### Testing Data ##############################


	TestData = TestInfo[:,1:]
	Testseq_length=[]
	for i in range(TestData.shape[0]):
		seq=0
		for j in range (0,TestData.shape[1],int(numberOfFeatures)):
			if(TestData[i,j]!=0):
				seq+=1
		Testseq_length.append(seq)
	Testseq_length = np.array(Testseq_length)
	TestData =  TestData.reshape((int(TestData.shape[0]), 19 ,int(numberOfFeatures) ))
	#print TrainData.shape , CompleteTrainOutput.shape , Trainseq_length.shape , TestData.shape , Testseq_length.shape
	RID =TestInfo[:,0]

	if (Validation):
		rid = TrainInfo[:,0]
		unqRID = np.unique(rid)
		TrainRID , TestRID = train_test_split(
	    unqRID, test_size=0.2, random_state=42)
		testingIndex=[]
		trainingIndex=[]
		for i in range(TrainInfo.shape[0]):
			if(TrainInfo[i,0] in TrainRID):
				trainingIndex.append(i)
			if(TrainInfo[i,0] in TestRID):
				testingIndex.append(i)
		splits =  np.array([ trainingIndex ,  testingIndex])

		train_lstm("DX" ,TrainData, CompleteTrainOutput,Trainseq_length, TestData , Testseq_length,RID , numberOfFeaturesDxTest ,  numberOfFeaturesCDRSBTest  ,  numberOfFeaturesMMSETest ,
			splits=splits , CDRSB_ = TrainOutputCDRSB, MMSE_ = TrainOutputMMSE)
	else:
		train_lstm("DX" ,TrainData, CompleteTrainOutput,Trainseq_length, TestData , Testseq_length,RID  , numberOfFeaturesDxTest ,  numberOfFeaturesCDRSBTest  ,  numberOfFeaturesMMSETest
		, CDRSB_ = TrainOutputCDRSB, MMSE_ = TrainOutputMMSE )


def  getAUC2 (true , output):
	correct = 0
	for i in range(true.shape[0]):
		#print true[i] , output[i], "true", np.argmax(true[i])+1 ,"predicted" , np.argmax(output[i])+1
		#print "true", np.argmax(true[i])+1 ,"predicted" , np.argmax(output[i])+1
		if(np.argmax(true[i]) ==    np.argmax(output[i])):
			correct+=1
	return float(correct)/true.shape[0]

def  getAUC (true , output):
	correct = 0
	for i in range(true.shape[0]):
		#print true[i] , output[i], np.argmax(output[i]) , np.argmax(true[i])
		if(np.argmax(true[i]) ==    np.argmax(output[i])):
			correct+=1
	return float(correct)/true.shape[0]

def getPrediction(output):
	return np.argmax(output)+1

def train_lstm(TargetName ,X_, Y_,Trainseq_length, TestData , Testseq_length,RID, 
				numberOfFeaturesDxTest ,  numberOfFeaturesCDRSBTest  ,  numberOfFeaturesMMSETest ,splits=None, CDRSB_=None ,  MMSE_=None,
				learning_rate = 0.05 ,n_neurons=64, n_layers = 2 , alpha=0.15,n_epochs=200,dropoutKeepProb=1.0):
				# learning_rate = 0.05 ,n_neurons=64, n_layers = 2 , alpha=0.2,n_epochs=200 , dropoutKeepProb=0.8):
	#Number of steps is the number of timeseries in this case its 5 2-1,3-2,4-3,5-4 and 6-5
	n_steps = Y_.shape[1]
	n_inputs = X_.shape[2]
	n_outputs = Y_.shape[2]
	num_classes=3

	seq_length = tf.placeholder(tf.int32, [None])
	X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
	y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])
	CDRSB = tf.placeholder(tf.float32, [None])
	MMSE = tf.placeholder(tf.float32, [None])
	



	keep_prob = tf.placeholder_with_default(dropoutKeepProb, [])


	#create lstm with 2 layers each with 64 hidden units
	lstm_cells = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)
	              for layer in range(n_layers)]


	multi_cell = tf.contrib.rnn.OutputProjectionWrapper(
		tf.contrib.rnn.MultiRNNCell(lstm_cells),output_size=3)

	outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32, sequence_length=seq_length)
	# Add dropout, as the model otherwise quickly overfits
	rnn_outputs = tf.nn.dropout(outputs, keep_prob)
	shape = rnn_outputs.get_shape().as_list() 
	last_rnn_output = last_relevant(rnn_outputs, seq_length)
	rnn_outputs = tf.reshape(rnn_outputs, [-1, n_outputs])  

	

	with tf.variable_scope('softmax'):
	    W = tf.get_variable('W', [n_outputs, num_classes])
	    b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

	logits = tf.matmul(rnn_outputs, W) + b
	logits = tf.reshape(logits, [-1,shape[1]  , shape[2]])  
	

	last_y= last_relevant(y, seq_length)
	last_logits = last_relevant(logits, seq_length)
	preds = tf.nn.softmax(last_logits)

	xentropyloss = 	loss_function(y, logits,seq_length , alpha)
	#print "last_rnn_output" , last_rnn_output
	MAECDRSB = tf.reduce_mean(tf.abs(last_rnn_output[:,-2] -CDRSB ))
	MAEMMSE = tf.reduce_mean(tf.abs(last_rnn_output[:,-1] -MMSE ))

	loss = xentropyloss + 0.1*MAECDRSB+0.05*MAEMMSE
	

	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	training_op = optimizer.minimize(loss)
	# xentropy =loss

	# xentropy = -tf.reduce_sum(lasty * tf.log(lastprediction))

	 
	init = tf.global_variables_initializer()
	saver = tf.train.Saver()

	if(splits is not None):
		X_train = X_[splits[0]]
		X_test =  X_[splits[1]]
		Y_train = Y_[splits[0]]
		Y_test =  Y_[splits[1]]
		seq_Train = Trainseq_length[splits[0]]
		seq_Test =  Trainseq_length[splits[1]]
		CDRSB_Train= CDRSB_[splits[0]]
		CDRSB_Test= CDRSB_[splits[1]]
		MMSE_Train= MMSE_[splits[0]]
		MMSE_Test= MMSE_[splits[1]]

		with tf.Session() as sess:
			init.run()
			for epoch in range(n_epochs):

				X_train , Y_train , seq_Train,CDRSB_Train , MMSE_Train = unison_shuffled_copiesFull(X_train,Y_train,seq_Train,CDRSB_Train , MMSE_Train )
				X_test , Y_test , seq_Test,CDRSB_Test , MMSE_Test = unison_shuffled_copiesFull(X_test,Y_test,seq_Test,CDRSB_Test , MMSE_Test )
				sess.run(training_op, feed_dict={X: X_train, y: Y_train  ,  seq_length:seq_Train,CDRSB:CDRSB_Train , MMSE:MMSE_Train})
				
				last_y_ = last_y.eval(feed_dict={X: X_train, y: Y_train  ,  seq_length:seq_Train,CDRSB:CDRSB_Train , MMSE:MMSE_Train})
				prediction  = preds.eval( feed_dict={X: X_train, y: Y_train  ,  seq_length:seq_Train,CDRSB:CDRSB_Train , MMSE:MMSE_Train})
				acc_train = getAUC(last_y_ , prediction)
				
				train_CDRSB = MAECDRSB.eval( feed_dict={X: X_train, y: Y_train  ,  seq_length:seq_Train,CDRSB:CDRSB_Train , MMSE:MMSE_Train})
				train_mmse = MAEMMSE.eval( feed_dict={X: X_train, y: Y_train  ,  seq_length:seq_Train,CDRSB:CDRSB_Train , MMSE:MMSE_Train})

				last_y_Test = last_y.eval(feed_dict={X: X_test, y: Y_test,  seq_length:seq_Test,CDRSB:CDRSB_Test, MMSE:MMSE_Test})
				prediction_test = preds.eval(feed_dict={X: X_test, y: Y_test,  seq_length:seq_Test,CDRSB:CDRSB_Test, MMSE:MMSE_Test})
				acc_test = getAUC(last_y_Test , prediction_test)
				
				test_CDRSB = MAECDRSB.eval( feed_dict={X: X_test, y: Y_test  ,  seq_length:seq_Test,CDRSB:CDRSB_Test, MMSE:MMSE_Test})
				test_mmse = MAEMMSE.eval( feed_dict={X: X_test, y: Y_test  ,  seq_length:seq_Test,CDRSB:CDRSB_Test, MMSE:MMSE_Test})

				xentropy_train = xentropyloss.eval(feed_dict={X: X_train, y: Y_train  ,  seq_length:seq_Train,CDRSB:CDRSB_Train , MMSE:MMSE_Train})
				xentropy_test = xentropyloss.eval(feed_dict={X: X_test, y: Y_test,  seq_length:seq_Test , seq_length:seq_Test,CDRSB:CDRSB_Test, MMSE:MMSE_Test})

				print("Epoch", epoch, "Train AUC =", acc_train, "Test AUC=", acc_test ,  "Train xentropy =", xentropy_train, "Test xentropy=", xentropy_test)
				print("Train MAE CDRSB =", train_CDRSB, "Test MAE CDRSB =", test_CDRSB , "Train MAE MMSE =", train_mmse, "Test MAE MMSE =", test_mmse)
			# getAUC2(last_y_Test , prediction_test)
			saver.save(sess, "./"+TargetName +"LSTM_model_withValidation")
	else:
		with tf.Session() as sess:
			init.run()
			for epoch in range(n_epochs):
				X_, Y_, Trainseq_length , CDRSB_, MMSE_ = unison_shuffled_copiesFull(X_,Y_,Trainseq_length , CDRSB_, MMSE_)
				sess.run(training_op, feed_dict={X: X_, y: Y_  ,  seq_length:Trainseq_length,CDRSB:CDRSB_, MMSE:MMSE_})
				last_y_ = last_y.eval( feed_dict={X: X_, y: Y_  ,  seq_length:Trainseq_length,CDRSB:CDRSB_, MMSE:MMSE_})
				prediction  = preds.eval(  feed_dict={X: X_, y: Y_  ,  seq_length:Trainseq_length,CDRSB:CDRSB_, MMSE:MMSE_})
				acc_train = getAUC(last_y_ , prediction)
				xentropy_train = xentropyloss.eval( feed_dict={X: X_, y: Y_  ,  seq_length:Trainseq_length,CDRSB:CDRSB_, MMSE:MMSE_})
				
				train_CDRSB = MAECDRSB.eval(  feed_dict={X: X_, y: Y_  ,  seq_length:Trainseq_length,CDRSB:CDRSB_, MMSE:MMSE_})
				train_mmse = MAEMMSE.eval(  feed_dict={X: X_, y: Y_  ,  seq_length:Trainseq_length,CDRSB:CDRSB_, MMSE:MMSE_})

				print("Epoch", epoch, "Train AUC =", acc_train ,  "Train xentropy =", xentropy_train)
				print("Train MAE CDRSB =", train_CDRSB, "Train MAE MMSE =", train_mmse)
			saver.save(sess, "./"+TargetName +"LSTM_model")

			
			OutputPersistence = np.zeros((TestData.shape[0]*50, 4))

			for i in range(TestData.shape[0]):
				seq = np.array([Testseq_length[i]])
				tempseq = seq
				tempseq[0] = tempseq[0]-1
				temp = TestData[i].reshape(1, 19, n_inputs)
				label = TestData[i ,tempseq[0],0]
				Feaure1= TestData[i ,tempseq[0],1]
				temp[0 ,tempseq[0],0]=0
				temp[0 ,tempseq[0],1]=0
				last_rnn_output_=last_rnn_output.eval(feed_dict={X: temp , seq_length:tempseq})
				TestData[i ,tempseq[0],2] = last_rnn_output_.flatten()[1]
				TestData[i ,tempseq[0],2+numberOfFeaturesCDRSBTest] = last_rnn_output_.flatten()[2]
				indxbefore =  tempseq[0]-1
				for ind in range(1,n_inputs):
					if(ind!=1 and ind!=2 and ind!=2+numberOfFeaturesCDRSBTest):
						TestData[i ,tempseq[0],ind] = TestData[i ,indxbefore,ind]
				TestData[i ,tempseq[0],0] = label
				TestData[i ,tempseq[0],1] = Feaure1
				for j in range(50):
					X_batch = TestData[i].reshape(1, 19, n_inputs)
					y_pred = sess.run(preds, feed_dict={X: X_batch , seq_length:seq})
					last_rnn_output_=last_rnn_output.eval(feed_dict={X: X_batch , seq_length:seq})
					newlabel = getPrediction( y_pred.flatten())
					#print i, j ,  "newlabel", newlabel
					OutputPersistence[i*50+j,0] = RID[i]
					OutputPersistence[i*50+j,1] =  y_pred.flatten()[0]
					OutputPersistence[i*50+j,2] =  y_pred.flatten()[1]
					OutputPersistence[i*50+j,3] =  y_pred.flatten()[2]

					if(seq[0]<19):
						TestData[i ,seq[0],0] =newlabel
						TestData[i ,seq[0],1] =  TestData[i ,seq[0]-1,1]
						TestData[i ,seq[0],2] = last_rnn_output_.flatten()[1]
						TestData[i ,seq[0],2+numberOfFeaturesCDRSBTest] = last_rnn_output_.flatten()[2]
						indxbefore = seq[0] -1
						for ind in range(1,n_inputs):
							if(ind!=1 and ind!=2 and ind!=2+numberOfFeaturesCDRSBTest):
								TestData[i ,seq[0],ind] = TestData[i ,indxbefore,ind]
						seq[0]+=1

					else:
						np.roll(TestData[i], -1, axis=0)
						TestData[i] =np.roll(TestData[i], -1, axis=0)
						TestData[i , -1,0] = newlabel
						TestData[i ,-1,1] =  TestData[i ,-2,1]
						TestData[i ,-1,2] = last_rnn_output_.flatten()[1]
						TestData[i ,-1,2+numberOfFeaturesCDRSBTest] = last_rnn_output_.flatten()[2]
						for ind in range(1,n_inputs):
							if(ind!=1 and ind!=2 and ind!=2+numberOfFeaturesCDRSBTest):
								TestData[i , -1,ind] = TestData[i ,-2,ind]
				

			df = DataFrame(OutputPersistence,columns=["RID" , "CN" , "MCI" , "AD"])
			df.to_csv('/Users/Tim/Desktop/Alzheimer/ForecastProcessed_data/DXLeaderboradOutputPersistence.csv',index=False)



