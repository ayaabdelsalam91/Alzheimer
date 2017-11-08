from pandas import DataFrame
from pandas import read_csv
import random
import math
import numpy as np 
import tensorflow  as tf
from numpy import zeros, newaxis
from sklearn.model_selection import train_test_split
from scipy.ndimage.interpolation import shift

# np.set_printoptions(threshold='nan')

path = '/Users/Tim/Desktop/Alzheimer/ForecastProcessed_data/'


def unison_shuffled_copies(a, b, c):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]


def loss_function(y, outputs,seq_length , alpha):
    lasty = last_relevant(y, seq_length)
    lastOutput = last_relevant(outputs, seq_length)
    Lastloss = tf.reduce_sum(tf.abs(lasty -lastOutput))
    AllExceptloss =  tf.reduce_sum(tf.abs(y -outputs)) - Lastloss
    return tf.reduce_mean(AllExceptloss*alpha +Lastloss*(1-alpha))



def last_relevant(output, length):
    batch_size = tf.shape(output)[0]

    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    print(batch_size , max_length , out_size)
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant



def get_interval(MAE):
	return np.percentile(MAE,25),np.percentile(MAE,75)

def LSTM(TargetName , TrainInputFile , TestInputFile, Validation=False):

	Traindata = path+TrainInputFile+'.csv'
	Traindataframe = read_csv(Traindata, names=None)
	TrainInfo = Traindataframe.values
	Testdata = path+TestInputFile+'.csv'
	Testdataframe = read_csv(Testdata, names=None)
	TestInfo =  Testdataframe.values
	numberOfFeatures=0
	for i in range (1, TrainInfo.shape[1]):
		if("0" in Traindataframe.columns.values[i] or "1" in Traindataframe.columns.values[i]):
			numberOfFeatures+=1
		else:
			break


	if Validation:
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

	######################### For Training ####################

	np.random.shuffle(TrainInfo)

	TrainOutput = TrainInfo[:,-1]

	TrainData = TrainInfo[:,1:-1]
	Trainseq_length=[]
	for i in range(TrainData.shape[0]):
		seq=0
		# print TrainData[i,:]
		for j in range (0,TrainData.shape[1],numberOfFeatures):
			if(TrainData[i,j]!=0):
				seq+=1
		Trainseq_length.append(seq)

	Trainseq_length = np.array(Trainseq_length)

	CompleteTrainOutput = np.zeros((TrainOutput.shape[0],20,1))

	for i in range (TrainOutput.shape[0]):
		for j in range(Trainseq_length[i]):
			CompleteTrainOutput[i,j,0] = TrainOutput[i]

	TrainData =  TrainData.reshape((TrainData.shape[0], 20 ,numberOfFeatures ))
	######################### For Testing ####################

	TestData = TestInfo[:,1:-numberOfFeatures]
	Testseq_length=[]
	for i in range(TestData.shape[0]):
		seq=0
		for j in range (0,TestData.shape[1],numberOfFeatures):
			if(TestData[i,j]!=0):
				seq+=1
		Testseq_length.append(seq)
	Testseq_length = np.array(Testseq_length)
	TestData =  TestData.reshape((TestData.shape[0], 20 ,numberOfFeatures ))
	# print TrainData.shape , CompleteTrainOutput.shape , Trainseq_length.shape , TestData.shape , Testseq_length.shape
	RID =TestInfo[:,0]
	
	if(Validation):
		train_lstm(TargetName ,TrainData, CompleteTrainOutput,Trainseq_length,TestData , Testseq_length,RID,splits)
	else:
		train_lstm(TargetName ,TrainData, CompleteTrainOutput,Trainseq_length,TestData , Testseq_length,RID)
	# test_lstm(TargetName ,TestData , Testseq_length , RID)







def train_lstm(TargetName ,X_, Y_,Trainseq_length, TestData , Testseq_length,RID ,splits=None,
	learning_rate = 0.05 ,n_neurons=64, n_layers = 2 , alpha=0.15,n_epochs=2,dropoutKeepProb=1.0):
	#Number of steps is the number of timeseries in this case its 5 2-1,3-2,4-3,5-4 and 6-5
	n_steps = Y_.shape[1]
	n_inputs = X_.shape[2]
	n_outputs = Y_.shape[2]

	#create a placeholder for input and output
	X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
	y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])
	seq_length = tf.placeholder(tf.int32, [None])
	keep_prob = tf.placeholder_with_default(dropoutKeepProb, [])

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


	outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32, sequence_length=seq_length)
	rnn_outputs = tf.nn.dropout(outputs, keep_prob)
	loss = loss_function(y, rnn_outputs,seq_length,alpha)
	y_Last = last_relevant(y, seq_length)
	outputs_Last = last_relevant(rnn_outputs, seq_length)
	accuracy = tf.reduce_mean(tf.abs(y_Last-outputs_Last))
	MAE = tf.abs(y_Last-outputs_Last)
	#optimizer optimzes loss
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	training_op = optimizer.minimize(loss)

	init = tf.global_variables_initializer()
	saver = tf.train.Saver()

	if(splits is not None):
		X_train = X_[splits[0]]
		X_test =  X_[splits[1]]
		Y_train = Y_[splits[0]]
		Y_test =  Y_[splits[1]]
		seq_Train = Trainseq_length[splits[0]]
		seq_Test =  Trainseq_length[splits[1]]
		with tf.Session() as sess:
			init.run()
			for epoch in range(n_epochs):

				X_train , Y_train, seq_Train = unison_shuffled_copies(X_train,Y_train, seq_Train)
				# X_test , Y_test,seq_Test = unison_shuffled_copies(X_test,Y_test,seq_Test)

				sess.run(training_op, feed_dict={X: X_train, y: Y_train  ,  seq_length:seq_Train})
				acc_train = accuracy.eval(feed_dict={X: X_train, y: Y_train  ,  seq_length:seq_Train})
				acc_test = accuracy.eval(feed_dict={X: X_test, y: Y_test,  seq_length:seq_Test})
				acc_testy_Last = y_Last.eval(feed_dict={X: X_test, y: Y_test,  seq_length:seq_Test})
				acc_testACC = outputs_Last.eval(feed_dict={X: X_test, y: Y_test,  seq_length:seq_Test})

				print("Epoch", epoch, "Train MAE =", acc_train, "Test MAE =", acc_test)
				#print(Y_train)
				#print(X_train,np.count_nonzero(seq_Test),np.size(seq_Test), (seq_Test==0).sum(), seq_Train)
				#print (acc_testy_Last ,acc_testACC )
			MAE_values = MAE.eval(feed_dict={X: X_train, y: Y_train  ,  seq_length:seq_Train})
			interval_25 , interval_75 = get_interval(MAE_values)
			MAE = np.mean(MAE_values)
			interval_25, interval75 = MAE - interval_25, interval_75 - MAE
			print(interval_25,interval_75)
			saver.save(sess, "./"+TargetName +"LSTM_model_withValidation")

			OutputPersistence = np.zeros((TestData.shape[0] * 50, 4))
			for i in range(TestData.shape[0]):
				seq = np.array([Testseq_length[i]])
				for j in range(50):
					X_batch = TestData[i].reshape(1, 20, n_inputs)
					y_pred = sess.run(outputs_Last, feed_dict={X: X_batch, seq_length: seq})
					# print i , j , y_pred.flatten()[0] , interval_25 , interval_75
					OutputPersistence[i * 50 + j, 0] = RID[i]
					OutputPersistence[i * 50 + j, 1] = y_pred.flatten()[0]
					if ((y_pred.flatten()[0] - interval_25) > 0):
						OutputPersistence[i * 50 + j, 2] = y_pred.flatten()[0] - interval_25
					else:
						OutputPersistence[i * 50 + j, 2] = y_pred.flatten()[0]
					OutputPersistence[i * 50 + j, 3] = y_pred.flatten()[0] + interval_75

					if (seq[0] < 20):
						TestData[i, seq[0], 0] = OutputPersistence[i * 50 + j, 1]
						for ind in range(1, n_inputs):
							TestData[i, seq[0], ind] = TestData[i, seq[0] - 1, ind]
						seq[0] += 1

					else:
						np.roll(TestData[i], -1, axis=0)
						TestData[i] = np.roll(TestData[i], -1, axis=0)
						TestData[i, -1, 0] = OutputPersistence[i * 50 + j, 1]
						for ind in range(1, n_inputs):
							TestData[i, -1, ind] = TestData[i, -2, ind]
			df = DataFrame(OutputPersistence, columns=["RID", TargetName, "-25", "+75"])
			df.to_csv(
				path + TargetName + 'LeaderboardOutputPersistence_Validation.csv',
				index=False)
	else:
		with tf.Session() as sess:
			init.run()
			for epoch in range(n_epochs):
				X_, Y_,Trainseq_length = unison_shuffled_copies(X_,Y_,Trainseq_length)
				#X_, Y_ = unison_shuffled_copies(X_, Y_)

				sess.run(training_op, feed_dict={X: X_, y: Y_  ,  seq_length:Trainseq_length})
				acc_train = accuracy.eval(feed_dict={X: X_, y: Y_  ,  seq_length:Trainseq_length})
				MAE_values = MAE.eval(feed_dict={X: X_, y: Y_  ,  seq_length:Trainseq_length})
				interval_25 , interval_75 = get_interval(MAE_values)
				print(interval_25,interval_75)
				trueMAE = np.mean(MAE_values)
				print("MAE ", trueMAE)
				interval_25,interval_75 = trueMAE-interval_25,interval_75-trueMAE
				print("Epoch", epoch, "Train MAE =", acc_train ,"interval_25" ,interval_25 ,"interval_75" , interval_75 )
			saver.save(sess, "./"+TargetName +"LSTM_model")

			# OutputZero = np.zeros((TestData.shape[0]*50, 4))
			# for i in range(2):
			# 	seq = np.array([Testseq_length[i]])
			# 	for j in range(50):
			# 		print TestData[i]
			# 		X_batch = TestData[i].reshape(1, 20, n_inputs)
			# 		y_pred = sess.run(outputs_Last, feed_dict={X: X_batch , seq_length:seq})
			# 		print i , j ,y_pred , interval_25 , interval_75
			# 		OutputZero[i*50+j,0] = RID[i]
			# 		OutputZero[i*50+j,1] =  y_pred.flatten()[0]
			# 		OutputZero[i*50+j,2] =  y_pred.flatten()[0] - interval_25
			# 		OutputZero[i*50+j,3] =  y_pred.flatten()[0] + interval_75
					
			# 		if(seq[0]<20):
			# 			TestData[i , seq[0],0] =OutputZero[i*50+j,1] 
			# 			seq[0]+=1
			# 		else:
			# 			np.roll(TestData[i], -1, axis=0)
			# 			TestData[i] =np.roll(TestData[i], -1, axis=0)
			# 			TestData[i , -1,0] =  y_pred.flatten()[0]
			# 			for ind in range(1,n_inputs):
			# 				TestData[i , -1,ind] = 0

			#df = DataFrame(OutputZero,columns=["RID" , TargetName , "-25" , "+75"])
			#df.to_csv('/Users/Tim/Desktop/Alzheimer/ForecastProcessed_data/'+TargetName+ 'LeaderboradOutputZeroPadding.csv',index=False)
			
			OutputPersistence = np.zeros((TestData.shape[0]*50, 4))
			for i in range(TestData.shape[0]):
					seq = np.array([Testseq_length[i]])
					for j in range(50):
						X_batch = TestData[i].reshape(1, 20, n_inputs)
						y_pred = sess.run(outputs_Last, feed_dict={X: X_batch , seq_length:seq})
						#print i , j , y_pred.flatten()[0] , interval_25 , interval_75
						OutputPersistence[i*50+j,0] = RID[i]
						OutputPersistence[i*50+j,1] =  y_pred.flatten()[0]
						if( (y_pred.flatten()[0] - interval_25) >0):
							OutputPersistence[i*50+j,2] =  y_pred.flatten()[0] - interval_25
						else:
							OutputPersistence[i*50+j,2] =  y_pred.flatten()[0]
						OutputPersistence[i*50+j,3] =  y_pred.flatten()[0] + interval_75

						if(seq[0]<20):
							TestData[i ,seq[0],0] =OutputPersistence[i*50+j,1]
							for ind in range(1,n_inputs):
								TestData[i , seq[0],ind] = TestData[i ,seq[0]-1,ind]
							seq[0]+=1

						else:
							np.roll(TestData[i], -1, axis=0)
							TestData[i] =np.roll(TestData[i], -1, axis=0)
							TestData[i , -1,0] = OutputPersistence[i*50+j,1]
							for ind in range(1,n_inputs):
								TestData[i , -1,ind] = TestData[i ,-2,ind]
			df = DataFrame(OutputPersistence,columns=["RID" , TargetName , "-25" , "+75"])
			df.to_csv(path+TargetName+ 'LeaderboradOutputPersistence.csv',index=False)
	df = DataFrame(MAE_values,columns=["MAE"])
	df.to_csv(path+'TrainingMAE.csv',index=False)










