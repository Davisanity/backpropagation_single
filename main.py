from neural import Neural

# https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
# Implement from this website

def main():
	inputs = [(1,2),(2,3)]
	output = [(0.5,0.6),(0.3,0.7)]
	n = Neural(inputs,output,2,3,2)
	n.train_model(1500)
	# for itera in range(1000):
	# 	print("*****"+str(itera)+"*****")
	# 	for i in range(len(inputs)):
	# 		n.forword(n.w_ih,n.input[i],n.hidden,len(n.hidden)-1) #len()-1 because bias don't need to caculate
	# 		n.forword(n.w_ho,n.hidden,n.output,len(n.output))
	# 		n.e_total=n.caculate_total_error(i)
	# 		n.back_propagete_error(i)
	# 		# print(n.w_ho)
	# 		# print(n.w_ih)
	# 		print("OUTPUT:",n.output)
	# 	print("ERROR:",n.e_total)

	# 	print ("updated w_h",n.w_h)
	# 	print ("updated w_o",n.w_o)
	# # print ("updated w_ih",n.w_ih)
	# # print ("updated w_h",n.w_h)
	# # print ("updated w_ho",n.w_ho)
	# # print ("updated w_o",n.w_o)
	# print ("output",n.output)



if __name__ == "__main__":
    main()
