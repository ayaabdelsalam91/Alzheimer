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


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]




def loss_function(y, outputs,seq_length , alpha):
	lasty = last_relevant(y, seq_length)
	lastOutput = last_relevant(outputs, seq_length)
	Lastloss = tf.reduce_sum(tf.abs(lasty -lastOutput))
	AllExceptloss =  tf.reduce_sum(tf.abs(y -outputs)) - Lastloss

 	return tf.reduce_mean(AllExceptloss*alpha +Lastloss*(1-alpha) )



def last_relevant(output, length):
  batch_size = tf.shape(output)[0]

  max_length = tf.shape(output)[1]
  out_size = int(output.get_shape()[2])
  print batch_size , max_length , out_size
  index = tf.range(0, batch_size) * max_length + (length - 1)
  flat = tf.reshape(output, [-1, out_size])
  relevant = tf.gather(flat, index)
  return relevant



def get_interval(MAE):
	return 0,0

def LSTM(TargetName , TrainInputFile , TestInputFile, Validation=False):

	Traindata = '/Users/aya/Documents/Research/Alzheimer/tadpole_challenge/ForecastProcessed_data/'+TrainInputFile+'.csv'
	Traindataframe = read_csv(Traindata, names=None)
	TrainInfo = Traindataframe.values
	Testdata = '/Users/aya/Documents/Research/Alzheimer/tadpole_challenge/ForecastProcessed_data/'+TestInputFile+'.csv'
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
	learning_rate = 0.0000001 ,n_neurons=30, n_layers = 2 , alpha=0.2,n_epochs=10):
	#Number of steps is the number of timeseries in this case its 5 2-1,3-2,4-3,5-4 and 6-5
	n_steps = Y_.shape[1]
	n_inputs = X_.shape[2]
	n_outputs = Y_.shape[2]

	#create a placeholder for input and output
	X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
	y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])
	seq_length = tf.placeholder(tf.int32, [None])

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
	
	loss = loss_function(y, outputs,seq_length,alpha)
	y_Last = last_relevant(y, seq_length)
	outputs_Last = last_relevant(outputs, seq_length)
	outputs_Lastvariable = tf.identity(outputs_Last, name='outputs_Last')
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
				X_test , Y_test = unison_shuffled_copies(X_test,Y_test)
				sess.run(training_op, feed_dict={X: X_train, y: Y_train  ,  seq_length:seq_Train})
				acc_train = accuracy.eval(feed_dict={X: X_train, y: Y_train  ,  seq_length:seq_Train})
				acc_test = accuracy.eval(feed_dict={X: X_test, y: Y_test,  seq_length:seq_Test})
				acc_testy_Last = y_Last.eval(feed_dict={X: X_test, y: Y_test,  seq_length:seq_Test})
				acc_testACC = outputs_Last.eval(feed_dict={X: X_test, y: Y_test,  seq_length:seq_Test})

				print("Epoch", epoch, "Train MAE =", acc_train, "Test MAE =", acc_test)
				# print (acc_testy_Last ,acc_testACC )
			MAE_values = MAE.eval(feed_dict={X: X_train, y: Y_train  ,  seq_length:seq_Train})
			iterval_25 , interval_75 = get_interval(MAE_values)
			saver.save(sess, "./"+TargetName +"LSTM_model_withValidation")
	else:
		with tf.Session() as sess:
			init.run()
			for epoch in range(2):
				X_, Y_ = unison_shuffled_copies(X_,Y_)
				sess.run(training_op, feed_dict={X: X_, y: Y_  ,  seq_length:Trainseq_length})
				acc_train = accuracy.eval(feed_dict={X: X_, y: Y_  ,  seq_length:Trainseq_length})
				print("Epoch", epoch, "Train MAE =", acc_train)
			MAE_values = MAE.eval(feed_dict={X: X_, y: Y_  ,  seq_length:Trainseq_length})
			iterval_25 , interval_75 = get_interval(MAE_values)
			saver.save(sess, "./"+TargetName +"LSTM_model")
			Output = np.zeros((TestData.shape[0] , 51))
			Output_25 = np.zeros((TestData.shape[0] , 51))
			Output_75 = np.zeros((TestData.shape[0] , 51))
			Output[:,0] = RID
			Output_25[:,0] = RID
			Output_75[:,0] = RID
			for i in range(TestData.shape[0]):
				seq = np.array([Testseq_length[i]])
				for j in range(50):
					X_batch = TestData[i].reshape(1, 20, n_inputs)
					y_pred = sess.run(outputs_Lastvariable, feed_dict={X: X_batch , seq_length:seq})
					Output[i,j+1] = TestData[i,seq[0]-1,0]  +  y_pred.flatten()[0]
					Output_25[i,j+1] = TestData[i,seq[0]-1,0]  +  y_pred.flatten()[0] -interval_25
					Output_75[i,j+1] = TestData[i,seq[0]-1,0]  +  y_pred.flatten()[0] +interval_75
					if(seq[0]<20):
						TestData[i , seq[0],0] =Output[i,j+1] 
						seq[0]+=1

					else:
						newValue = [0]*n_inputs
						newValue[0] =  y_pred.flatten()[0]
						np.roll(TestData[i], -1, axis=0)
						TestData[i] =np.roll(TestData[i], -1, axis=0)
						TestData[i , -1,0] = Output[i,j+1] 
						for ind in range(1,n_inputs):
							TestData[i , -1,ind] = 0



	df = DataFrame(MAE_values,columns=["MAE"])
	df.to_csv('/Users/aya/Documents/Research/Alzheimer/tadpole_challenge/ForecastProcessed_data/TrainingMAE.csv',index=False)








def test_lstm(TargetName ,TestData , Testseq_length,RID):
	Output = np.zeros((TestData.shape[0] , 51))
	Output[:,0] = RID
	n_steps = TestData.shape[1]
	n_inputs = TestData.shape[2]
	n_outputs = 1
	X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
	y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])
	seq_length = tf.placeholder(tf.int32, [None])

	print TestData.shape , Testseq_length.shape
	saver = tf.train.import_meta_graph(TargetName +'LSTM_model_withValidation.meta')

	with tf.Session() as sess:
		saver.restore(sess,  tf.train.latest_checkpoint('./'))
		sess.run(tf.global_variables_initializer())
		graph = tf.get_default_graph()
		outputs_Last = graph.get_tensor_by_name('outputs_Last:0')
		print outputs_Last

		sequence = [0.] * 20*4
		for iteration in range(300):
				X_batch = np.array(sequence[-20*4:]).reshape(1, 20, 4)
				seq = np.array([5])
				print X_batch ,  seq.shape
				y_pred = sess.run(outputs_Last, feed_dict={X: X_batch , seq_length:seq})
				print y_pred
	        
		# for i in range(1):
		# 	for j in range(50):
		# 		testinput = TestData[i].reshape(1,20,TestData.shape[2])
		# 		seq = Testseq_length[i].reshape(1,)
		# 		print "hello " , testinput.shape , seq.shape
		# 		y_pred =  sess.run(outputs_Last, feed_dict={X: testinput , seq_length: seq})
		# 		print y_pred



