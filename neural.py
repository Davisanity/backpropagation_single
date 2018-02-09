from numpy import random
import math
import numpy as np
import matplotlib.pyplot as plt

class Neural:
	#neural structure is 
	# o - o  o
	# o \ o  o
	#   \d o
	# O   O   -->bias
	def __init__(self,inputs,array_output,n_input,n_hidden,n_output):
		self.input = inputs
		for i in range(len(self.input)):
			self.input[i]=np.append(self.input[i],[1])
		self.hidden=np.zeros(n_hidden)
		self.hidden=np.append(self.hidden,[1])
		self.real_output = array_output
		self.output=np.zeros(n_output)

		self.w_ih=random.uniform(-1,2,size=(n_input+1,n_hidden)).round(decimals=3)
		self.w_ho=random.uniform(-1,2,size=(n_hidden+1,n_output)).round(decimals=3)
		
		self.learn = 0.5#learning rate
		#error here means delta in website
		self.sigma_o=np.zeros([n_output])
		self.sigma_h=np.zeros([n_hidden])
		self.e_total=0

	def forword(self,weight,input,out,n_o):	
		for j in range(len(weight[0])):
			for i in range(len(weight)):
				out[j]+=weight[i][j]*input[i]
		for j in range(n_o):
			out[j] = float('%.3f'%(self.sigmoid(out[j])))

	def sigmoid(self,sum):
		func = 1.0/(1+math.exp(sum*(-1)))
		return func

	def caculate_total_error(self,train_index):
		e_total=0
		for o in range(len(self.output)):
			e_total+=(self.real_output[train_index][o]-self.output[o])**2
		e_total=e_total/2
		return float('%.10f'%(e_total))	

	def transfer_derivative(self,output):
		return output * (1.0 - output)

	def back_propagete_error(self,train_index):
		# weight = weight + learning_rate * error * input
		# input for the output layer is a collection of outputs from the hidden layer.
		
		#update_hidden-> w_ho=w_ho - learn*delta
		#delta = [-(real[o]-out[o])]*[out[o](1-out[o])]*HIDDEN[h](???) = sigma * hidden
		s_o = self.sigma_o
		s_h = self.sigma_h
		r_o = self.real_output[train_index]
		out = self.output
		hidden=self.hidden
		inp = self.input[train_index]
		learn = self.learn
		w_ho = self.w_ho
		w_ih = self.w_ih
		for o in range(len(s_o)):
			s_o[o]=float('%.6f'%((out[o]-r_o[o])*self.transfer_derivative(out[o])))
		for h in range(len(s_h)):
			for o in range(len(s_o)):
				w_ho[h][o]=float('%.6f'%( w_ho[h][o]-learn*s_o[o]*hidden[h] ))
		# o<-o  E[o] means e[output0]+e[output1]
		# o \
		# o  o
		# O
		#here accumulate Etotal for every hidden
		#w_ih[i][h]=w_ih[i][h]-learn*(Ehidden_output_total)*transfer_derivative*input[i]
		Ehidden_output_total=np.zeros([len(hidden)])
		for h in range(len(hidden)):
			for o in range(len(out)):
				Ehidden_output_total[h]+=s_o[o]*w_ho[h][o]
			Ehidden_output_total[h]=float('%.6f'%(Ehidden_output_total[h]))
		# print (Ehidden_output_total)
		for i in range(len(inp)):
			for h in range(len(hidden)-1):
				w_ih[i][h]=float('%.6f'%( w_ih[i][h]-learn*Ehidden_output_total[h]*self.transfer_derivative(hidden[h])*inp[i] ))
			# len(hidden)-1 because hidden[-1] is bias,it's weight may not change(transfer_derivative will equals 0)

	#ERROR will become more bigger after train???
	def train_model(self,iter_times):
		pltx=np.arange(0,iter_times)
		plty=np.zeros(iter_times)
		for i in range(iter_times):
			temp_error=0
			self.e_total=0
			print("*****"+str(i)+"*****")
			for j in range(len(self.input)):
				self.forword(self.w_ih,self.input[j],self.hidden,len(self.hidden)-1) #len()-1 because bias don't need to caculate
				self.forword(self.w_ho,self.hidden,self.output,len(self.output))
				temp_error+=self.caculate_total_error(j)
				self.back_propagete_error(j)
				print("OUTPUT:",self.output)
			self.e_total=temp_error
			plty[i]=self.e_total
			print("ERROR:",self.e_total)
		#plot errors
		plt.subplot(211)
		plt.plot(pltx,plty)
		plt.title("Rough scatter")
		plt.ylabel("Total Error")

		plt.subplot(212)
		plt.plot(pltx,plty)
		plt.title("Accurate scatter")
		plt.ylabel("Total Error")
		plt.xlabel("iter times")
		plt.ylim((0,0.015))
		plt.show()


		# n.forword(n.w_ih,n.input,n.hidden,len(n.hidden)-1) #len()-1 because bias don't need to caculate
		# n.forword(n.w_ho,n.hidden,n.output,len(n.output))
		# n.e_total=n.caculate_total_error()
		# n.back_propagete_error()


