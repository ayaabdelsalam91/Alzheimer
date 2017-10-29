from pandas import DataFrame
from pandas import read_csv
import random
import math
import numpy as np 
import tensorflow  as tf
from tensorflow.examples.tutorials.mnist import input_data
from numpy import zeros, newaxis
from sklearn.model_selection import train_test_split
np.set_printoptions(threshold='nan')



def Map(input,type = "input"):
	if(type=="input"):
		if input==1:
			return[1,0,0]
		elif input==2:
			return[0,1,0]
		elif input==3:
			return[0,0,1]
	else:
		if input==1:
			return [[1,0,0] for i in range(5)]  
		elif input==2:
			return [[0,1,0] for i in range(5)]  
		elif input==3:
			return [[0,0,1] for i in range(5)]  

def loss_function( y_true,y_hat,n_steps , alpha=0.5):
	loss =tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_true[:,:-1,:] ,  logits = y_hat[:,:-1,:]))*alpha*(1./n_steps)
	loss += (1-alpha)*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_true[:,-1,:] ,  logits = y_hat[:,-1,:]))
	return loss


def mergeFiles():
	dataDX = '/Users/aya/Documents/Research/Alzheimer/tadpole_challenge/Proccessed_data/LSTMParsedTrainingData'+"DX"+'.csv'
	dataCDRSB = '/Users/aya/Documents/Research/Alzheimer/tadpole_challenge/Proccessed_data/LSTMParsedTrainingData'+"CDRSB"+'.csv'
	dataMMSE = '/Users/aya/Documents/Research/Alzheimer/tadpole_challenge/Proccessed_data/LSTMParsedTrainingData'+"MMSE"+'.csv'
	dataframeDX = read_csv(dataDX, names=None)
	dataframeCDRSB = read_csv(dataCDRSB, names=None)
	dataframeMMSE = read_csv(dataMMSE, names=None)
	size = dataframeDX.shape[1] + dataframeCDRSB.shape[1] + dataframeMMSE.shape[1] - 3*3 +1
	print size
	dataframeDX = dataframeDX.values
	dataframeCDRSB = dataframeCDRSB.values
	dataframeMMSE = dataframeMMSE.values
	input = np.zeros((dataframeMMSE.shape[0] ,4*5+1))
	numberOfFeaturesCDRSB = (dataframeCDRSB.shape[1]-3)/5
	numberOfFeaturesMMSE = (dataframeMMSE.shape[1]-3)/5
	for i in range(dataframeDX.shape[0]):
		for step  in range(5):
			input[i,0+step*4] = dataframeDX[i,2+step*2]
			input[i,1+step*4] = dataframeDX[i,2+step*2+1]
			input[i,2+step*4] = dataframeCDRSB[i,2+step*numberOfFeaturesCDRSB]
			input[i,3+step*4] = dataframeMMSE[i,2+step*numberOfFeaturesMMSE]
		input[i,-1] = dataframeDX[i,-1]
	cols=[]
	for i in range(5):
		cols.append("DX_"+str(i))
		cols.append("CDRSB_bl_"+str(i+1))
		cols.append("CDRSB_"+str(i+1))
		cols.append("MMSE_"+str(i+1))
	cols.append("Target")
	df = DataFrame(input , columns = cols)
	print df.shape
	df = df.drop_duplicates()
	print df.shape
	df.to_csv('/Users/aya/Documents/Research/Alzheimer/tadpole_challenge/Proccessed_data/LSTMParsedTrainingDataTempDX_andFeatures.csv',index=False)
	return df



def LSTMDX():
	dataframe =mergeFiles()
	info=dataframe.values
	print info.shape
	numberOfFeatures = (info.shape[1]-1)/5
	print (numberOfFeatures)
	rid = info[:,0]
	unqRID = np.unique(rid)
	TrainRID , TestRID = train_test_split(
    unqRID, test_size=0.2, random_state=42)
	testingIndex=[]
	trainingIndex=[]
	for i in range(info.shape[0]):
		if(info[i,0] in TrainRID):
			trainingIndex.append(i)
		if(info[i,0] in TestRID):
			testingIndex.append(i)
	X = info[:,:-1]
	print X.shape , numberOfFeatures
	NewX =X.reshape((X.shape[0], 5 ,numberOfFeatures ))
	Y=info[:,-1]
	NewY = []
	for i in range(Y.shape[0]):
		NewY.append(Map(Y[i],type = "target"))

	NewY = np.array(NewY)
	splits =  np.array([ trainingIndex ,  testingIndex])
	train_lstm(NewX, NewY ,splits, n_inputs=NewX.shape[2] , n_outputs=NewY.shape[2] , n_steps = NewX.shape[1])


def  getAUC (true , output, X_test):
	print "hereeeeeee", true.shape , output.shape
	correct = 0
	for i in range(true.shape[0]):
		print true[i] , output[i], np.argmax(output[i]) , np.argmax(true[i])
		if(np.argmax(true[i]) ==    np.argmax(output[i])):
			correct+=1
	return float(correct)/true.shape[0]



def train_lstm(X_, Y_ , splits,n_inputs=1,n_outputs=3 ,n_steps =5,n_neurons=30, n_layers = 2):
	#Number of steps is the number of timeseries in this case its 5 2-1,3-2,4-3,5-4 and 6-5
	learning_rate = 0.01
	#split training and testing data
	X_train = X_[splits[0]]
	X_test =  X_[splits[1]]
	Y_train = Y_[splits[0]]
	Y_test =  Y_[splits[1]]
	print n_inputs ,n_steps ,n_outputs


	X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
	y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])

	#create lstm with 2 layers each with 64 hidden units
	lstm_cells = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)
	              for layer in range(n_layers)]



	#make this LSTM recurrent 
	#At each time step we now have an output vector of size n_neurons.
	#But what we actually want is a single output value at each time step.
	#The simplest solution is to wrap the cell in an OutputProjectionWrapper.
	#A cell wrapper acts like a normal cell, proxying every method call to an underlying cell, but it also adds some functionality.
	#The OutputProjectionWrapper adds a fully connected layer of linear neurons (i.e., without any activation function)
	# on top of each output (but it does not affect the cell state).
	multi_cell = tf.contrib.rnn.OutputProjectionWrapper(
		tf.contrib.rnn.MultiRNNCell(lstm_cells),output_size=n_outputs)

	outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)
	
	loss = loss_function(y, outputs,n_steps)

	#optimizer optimzes loss
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	training_op = optimizer.minimize(loss)
	# correct = tf.nn.in_top_k(outputs, y, 1)
	#accuracy = tf.reduce_mean(xentropy)
	# loss
	softmax_out = tf.nn.softmax(outputs[:,-1,:], name=None)
	accuracy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y[:,-1,:],  logits = outputs[:,-1,:]))
                                                          # logits=logits)
	y_values = y[:,-1,:]
	outputs_values= softmax_out
	#accuracy = getAUC(y[:,-1,:] , outputs[:,-1,:])

	 
	init = tf.global_variables_initializer()
	n_epochs = 250
	outputs_ = 0
	ys_ = 0
	with tf.Session() as sess:
	    init.run()
	    for epoch in range(n_epochs):
	 
	        # print(X_train.shape)
	        # print(Y_train.shape)
	        print epoch

	        sess.run(training_op, feed_dict={X: X_train, y: Y_train})
	        acc_train = accuracy.eval(feed_dict={X: X_train, y: Y_train})
	        acc_test = accuracy.eval(feed_dict={X: X_test, y: Y_test})
	        outputs_ = outputs_values.eval(feed_dict={X: X_test, y: Y_test})
	        ys_ = y_values.eval(feed_dict={X: X_test, y: Y_test})
	        print("Epoch", epoch, "Train entropy =", acc_train, "Test entropy =", acc_test)
	    print getAUC(ys_ , outputs_ ,  X_test)
LSTMDX()