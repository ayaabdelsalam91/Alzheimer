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


def unison_shuffled_copies_Correct(a, b,c):
    assert len(a) == len(b)
    assert len(a) == len(c)
    p = np.random.permutation(len(a))
    return a[p], b[p] , c[p]


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


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

def LSTM(TargetName , TrainInputFile , TestInputFile, D3InputFile, Validation=False , DefaultValues = None , isBaseline = None):
	Traindata = '../Finaldata/'+TrainInputFile+'.csv'
	Traindataframe = read_csv(Traindata, names=None)
	TrainInfo = Traindataframe.values
	Testdata = '../Finaldata/'+TestInputFile+'.csv'
	Testdataframe = read_csv(Testdata, names=None)
	TestInfo =  Testdataframe.values
	D3data = '../Finaldata/'+D3InputFile+'.csv'
	D3dataframe = read_csv(D3data, names=None)
	D3Info =  D3dataframe.values
	numberOfFeatures=0
	for i in range (1, TrainInfo.shape[1]):
		if("0" in Traindataframe.columns.values[i] or "1" in Traindataframe.columns.values[i]):
			numberOfFeatures+=1
		else:
			break




	######################### For Training ####################

	np.random.shuffle(TrainInfo)

	TrainOutput = TrainInfo[:,-1]

	TrainData = TrainInfo[:,1:-1]
	Trainseq_length=[]
	toConsider=[]
	for i in range(TrainData.shape[0]):
		seq=0
		# print TrainData[i,:]
		for j in range (0,TrainData.shape[1],numberOfFeatures):
			if(TrainData[i,j]!=0):
				seq+=1
		if(seq>0):
			toConsider.append(i)
		Trainseq_length.append(seq)
	Trainseq_length = np.array(Trainseq_length)

	TrainOutput=TrainOutput[toConsider]
	Trainseq_length=Trainseq_length[toConsider]
	TrainData=TrainData[toConsider]
	CompleteTrainOutput = np.zeros((TrainOutput.shape[0],20,1))
	
	for i in range (TrainOutput.shape[0]):
		for j in range(Trainseq_length[i]):
			CompleteTrainOutput[i,j,0] = TrainOutput[i]

	
	if Validation:
		rid = TrainData[:,0]
		unqRID = np.unique(rid)
		TrainRID , TestRID = train_test_split(
	    unqRID, test_size=0.2, random_state=42)
		testingIndex=[]
		trainingIndex=[]
		for i in range(TrainData.shape[0]):
			if(TrainData[i,0] in TrainRID):
				trainingIndex.append(i)
			if(TrainData[i,0] in TestRID):
				testingIndex.append(i)
		splits =  np.array([ trainingIndex ,  testingIndex])

	TrainData =  TrainData.reshape((TrainData.shape[0], 20 ,numberOfFeatures ))
	print (TrainData.shape ,CompleteTrainOutput.shape )
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
	print( TrainData.shape , CompleteTrainOutput.shape , Trainseq_length.shape , TestData.shape , Testseq_length.shape)
	TestRID =TestInfo[:,0]


	D3Data = D3Info[:,1+numberOfFeatures:]
	D3seq_length=[]
	for i in range(D3Data.shape[0]):
		seq=0
		for j in range (0,D3Data.shape[1],numberOfFeatures):
			# print (D3Data[i,j:] , np.sum(D3Data[i,j:]))
			if(np.sum(D3Data[i,j:])!=0):
				seq+=1
		D3seq_length.append(seq)
	
	D3Data2 = D3Info[:,1:-numberOfFeatures]
	for i in range(len(D3seq_length)):
		if D3seq_length[i]==0:
			D3Data[i,:] = D3Data2[i,:]

	D3seq_length=[]
	for i in range(D3Data.shape[0]):
		seq=0
		for j in range (0,D3Data.shape[1],numberOfFeatures):
			# print (D3Data[i,j:] , np.sum(D3Data[i,j:]))
			if(np.sum(D3Data[i,j:])!=0):
				seq+=1
		D3seq_length.append(seq)

	D3seq_length = np.array(D3seq_length)


	D3Data =  D3Data.reshape((D3Data.shape[0], 20 ,numberOfFeatures ))
	D3RID =D3Info[:,0]
	if(Validation):
		train_lstm(TargetName ,TrainData, CompleteTrainOutput,Trainseq_length,TestData , Testseq_length,TestRID, D3Data , D3seq_length,D3RID,splits = splits)
	else:
		train_lstm(TargetName ,TrainData, CompleteTrainOutput,Trainseq_length,TestData , Testseq_length,TestRID,D3Data , D3seq_length,D3RID, DefaultValues = DefaultValues, isBaseline = isBaseline )
	# test_lstm(TargetName ,TestData , Testseq_length , RID)




def printstuff(x,y):
	for i in range(x.shape[0]):
		print (x[i] , y[i])


def train_lstm(TargetName ,X_, Y_,Trainseq_length, TestData , Testseq_length,RID, D3Data , D3seq_length,D3RID ,splits=None,
	learning_rate = 0.0003,n_neurons=64, n_layers = 2 , alpha=0.25,n_epochs=1200, DefaultValues = None  , isBaseline =None):
	print (TargetName)
	print(X_.shape, Y_.shape,Trainseq_length.shape)
	#Number of steps is the number of timeseries in this case its 5 2-1,3-2,4-3,5-4 and 6-5
	n_steps = Y_.shape[1]
	n_inputs = X_.shape[2]
	n_outputs = Y_.shape[2]
	print ("n_inputs" ,  n_inputs)
	print ("n_steps" ,  n_steps)
	print ("n_outputs" ,  n_outputs)



	#create a placeholder for input and output
	X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
	y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])
	seq_length = tf.placeholder(tf.int32, [None])
	keep_prob = tf.placeholder(tf.float32)
	

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

				X_train , Y_train = unison_shuffled_copies(X_train,Y_train)
				#X_train , Y_train,seq_Train = unison_shuffled_copies_Correct(X_train,Y_train ,seq_Train )


				sess.run(training_op, feed_dict={X: X_train, y: Y_train  ,  seq_length:seq_Train ,keep_prob:1.0})
				acc_train = accuracy.eval(feed_dict={X: X_train, y: Y_train  ,  seq_length:seq_Train ,keep_prob:1.0})
				acc_test = accuracy.eval(feed_dict={X: X_test, y: Y_test,  seq_length:seq_Test ,keep_prob:1.0})
				acc_testy_Last = y_Last.eval(feed_dict={X: X_test, y: Y_test,  seq_length:seq_Test ,keep_prob:1.0})
				acc_testACC = outputs_Last.eval(feed_dict={X: X_test, y: Y_test,  seq_length:seq_Test ,keep_prob:1.0})


				print("Epoch", epoch, "Train MAE =", acc_train, "Test MAE =", acc_test)
				#print(Y_train)
				#print(X_train,np.count_nonzero(seq_Test),np.size(seq_Test), (seq_Test==0).sum(), seq_Train)
				
			# printstuff(acc_testy_Last ,acc_testACC )
			MAE_values = MAE.eval(feed_dict={X: X_train, y: Y_train  ,  seq_length:seq_Train,keep_prob:1.0})
			interval_25 , interval_75 = get_interval(MAE_values)
			trueMAE = np.mean(MAE_values)
			interval_25,interval_75 = trueMAE-interval_25,interval_75-trueMAE
			saver.save(sess, "./"+TargetName +"LSTM_model_withValidation")
	else:
		with tf.Session() as sess:
			init.run()
			for epoch in range(n_epochs):
				print (X_.shape,Y_.shape , Trainseq_length.shape)
				X_, Y_ = unison_shuffled_copies(X_,Y_)
				#X_ , Y_,Trainseq_length = unison_shuffled_copies_Correct(X_,Y_ ,Trainseq_length )
	
				sess.run(training_op, feed_dict={X: X_, y: Y_  ,  seq_length:Trainseq_length ,keep_prob:1.0})
				acc_train = accuracy.eval(feed_dict={X: X_, y: Y_  ,  seq_length:Trainseq_length ,keep_prob:1.0})
				MAE_values = MAE.eval(feed_dict={X: X_, y: Y_  ,  seq_length:Trainseq_length ,keep_prob:1.0})
				interval_25 , interval_75 = get_interval(MAE_values)
				trueMAE = np.median(MAE_values)
				print("Epoch", epoch, "Train MAE =", acc_train ,"interval_25" ,interval_25 ,"interval_75" , interval_75 , "trueMAE" , trueMAE )
				interval_25,interval_75 = trueMAE-interval_25,interval_75-trueMAE
				
				#if(acc_train<0.002):
				#	break
			saver.save(sess, "./"+TargetName +"LSTM_model")

			OutputPersistence = np.zeros((TestData.shape[0]*50, 4))
			for i in range(TestData.shape[0]):
					seq = np.array([Testseq_length[i]])
					for j in range(50):
						X_batch = TestData[i].reshape(1, 20, n_inputs)
						y_pred = sess.run(outputs_Last, feed_dict={X: X_batch , seq_length:seq ,keep_prob:1.0})
						# print (i , j , y_pred.flatten()[0] , interval_25 , interval_75)
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
							if(DefaultValues==None):
								for ind in range(1,n_inputs):
										# print(TestData[i , -1,ind])
										TestData[i , -1,ind] = TestData[i ,-2,ind]
							else:
								# print("In ELSE")
								for ind in range(1,n_inputs):
										if(isBaseline[ind]==1):
											TestData[i , -1,ind] = TestData[i ,-2,ind]
										else:
											TestData[i ,-1,ind] =  DefaultValues[ind]
			df = DataFrame(OutputPersistence,columns=["RID" , TargetName , "-25" , "+75"])
			df.to_csv('../Finaldata/'+TargetName+ 'OutputPersistence.csv',index=False)




			D3Persistence = np.zeros((D3Data.shape[0]*50, 4))
			for i in range(D3Data.shape[0]):
					seq = np.array([D3seq_length[i]])
					for j in range(50):
						X_batch = D3Data[i].reshape(1, 20, n_inputs)
						y_pred = sess.run(outputs_Last, feed_dict={X: X_batch , seq_length:seq ,keep_prob:1.0})
						# print (i , j , y_pred.flatten()[0] , interval_25 , interval_75)
						D3Persistence[i*50+j,0] = RID[i]
						D3Persistence[i*50+j,1] =  y_pred.flatten()[0]
						if( (y_pred.flatten()[0] - interval_25) >0):
							D3Persistence[i*50+j,2] =  y_pred.flatten()[0] - interval_25
						else:
							D3Persistence[i*50+j,2] =  y_pred.flatten()[0]
						D3Persistence[i*50+j,3] =  y_pred.flatten()[0] + interval_75

						if(seq[0]<20):
							D3Data[i ,seq[0],0] =D3Persistence[i*50+j,1]
							for ind in range(1,n_inputs):
								D3Data[i , seq[0],ind] = D3Data[i ,seq[0]-1,ind]
							seq[0]+=1

						else:
							np.roll(D3Data[i], -1, axis=0)
							D3Data[i] =np.roll(D3Data[i], -1, axis=0)
							D3Data[i , -1,0] = D3Persistence[i*50+j,1]
							if(DefaultValues==None):
								for ind in range(1,n_inputs):
										# print(D3Data[i , -1,ind])
										D3Data[i , -1,ind] = D3Data[i ,-2,ind]
							else:
								# print("In ELSE")
								for ind in range(1,n_inputs):
										if(isBaseline[ind]==1):
											D3Data[i , -1,ind] = D3Data[i ,-2,ind]
										else:
											D3Data[i ,-1,ind] =  DefaultValues[ind]
			df = DataFrame(D3Persistence,columns=["RID" , TargetName , "-25" , "+75"])
			df.to_csv('../Finaldata/'+TargetName+ 'D3Persistence.csv',index=False)



	df = DataFrame(MAE_values,columns=["MAE"])
	df.to_csv('../Finaldata/'+TargetName+'TrainingMAE.csv',index=False)
