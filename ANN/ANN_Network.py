import numpy as np
import time
import matplotlib.pyplot as plt

from ANN_Project_Assets.Loading_Datasets import Loading_Datasets


class ANN_Network:

    def __init__(self):
        self.train_set, self.test_set = self.load_data()


    def load_data(self):
        train_plk_path = "ANN_Project_Assets"
        test_plk_path = "ANN_Project_Assets"
        # read plk files and load
        loading_datasets = Loading_Datasets(train_plk_path, test_plk_path)
        train_set = loading_datasets.train_set
        test_set = loading_datasets.test_set
        return train_set,test_set


    def cost(self, a, y):
        cost = (1 / y.shape[1]) * np.sum(np.power(np.subtract(a, y), 2))
        cost = np.squeeze(cost)
        return cost


    def derivative_cost(self,a,y):
        return 2 * (a - y)


    def update_weights(self, weights, learning_rate, x):
        new_weights = weights - learning_rate * x
        return new_weights


    def initialize_layer_parameters(self, HL_x1_neurons, HL_x2_neurons):
        HL_weights = np.random.normal(size=(HL_x1_neurons, HL_x2_neurons))
        HL_bias = np.zeros((HL_x1_neurons, 1))
        return HL_weights,HL_bias


    def initialize_layers(self, layers):
        layers_parameters = list()
        for l in range(0,len(layers)-1):
            HLp_neurons = layers[l]
            HLn_neurons = layers[l+1]
            HLp_HLn_weights,HLp_HLn_bias = self.initialize_layer_parameters(HLn_neurons, HLp_neurons)
            layers_parameters.append(tuple([HLp_HLn_weights, HLp_HLn_bias]))
            # print(layers_parameters[l][0].shape)
        return layers_parameters


    def sigmoid(self,inpt):
        return 1 / (1 + np.exp(-inpt))


    def derivative_sigmoid(self,x):
        return x * (1 - x)


    def linear_forward(self, next_w, a, next_b):
        next_a = self.sigmoid(next_w @ a + next_b)
        return next_a


    def linear_forward_calculation(self, data_input, layers_param): 
        a_list = list()
        a = data_input
        # layers_param:(w,b)
        for l in range(0,len(layers_param)):
            w = layers_param[l][0]
            b = layers_param[l][1]
            a = self.sigmoid((w @ a + b))
            a_list.append(a)
        return a_list


    def initialize_gradient_matrix(self,HL_x1_neurons, HL_x2_neurons):
        grad_w = np.zeros(size=(HL_x1_neurons, HL_x2_neurons))
        grad_b = np.zeros((HL_x1_neurons, 1))
        return grad_w,grad_b
        

    def backpropagation_calculation(self, image, label, a_list, grad_l_param, l_param ,layers):
        W1 = l_param[0][0]
        W2 = l_param[1][0]
        W3 = l_param[2][0]

        grad_W1 = grad_l_param[0][0]
        grad_W2 = grad_l_param[1][0]
        grad_W3 = grad_l_param[2][0]

        grad_b1 = grad_l_param[0][1]
        grad_b2 = grad_l_param[1][1]
        grad_b3 = grad_l_param[2][1]

        a1 = a_list[0]
        a2 = a_list[1]
        a3 = a_list[2]

        # ---- Last layer
        # weight
        for j in range(grad_W3.shape[0]):
            for k in range(grad_W3.shape[1]):
                grad_W3[j, k] += self.derivative_cost(a3[j, 0],label[j, 0]) * self.derivative_sigmoid(a3[j, 0]) * a2[k, 0]
        # bias
        for j in range(grad_b3.shape[0]):
                grad_b3[j, 0] += self.derivative_cost(a3[j, 0],label[j, 0]) * self.derivative_sigmoid(a3[j, 0])
        
        # ---- 3rd layer
        # activation
        delta_3 = np.zeros((60, 1))
        for k in range(60):
            for j in range(4):
                delta_3[k, 0] += self.derivative_cost(a3[j, 0],label[j, 0]) * self.derivative_sigmoid(a3[j, 0]) * W3[j, k]
        # weight
        for k in range(grad_W2.shape[0]):
            for m in range(grad_W2.shape[1]):
                grad_W2[k, m] += delta_3[k, 0] * self.derivative_sigmoid(a2[k,0]) * a1[m, 0]
        # bias
        for k in range(grad_b2.shape[0]):
                grad_b2[k, 0] += delta_3[k, 0] * self.derivative_sigmoid(a2[k,0])

        # ---- 2nd layer
        # activation
        delta_2 = np.zeros((150, 1))
        for m in range(150):
            for k in range(60):
                delta_2[m, 0] += delta_3[k, 0] * self.derivative_sigmoid(a2[k,0]) * W2[k, m]
        # weight
        for m in range(grad_W1.shape[0]):
            for v in range(grad_W1.shape[1]):
                grad_W1[m, v] += delta_2[m, 0] * self.derivative_sigmoid(a1[m,0]) * image[v, 0]
        # bias
        for m in range(grad_b1.shape[0]):
                grad_b1[m, 0] += delta_2[m, 0] * self.derivative_sigmoid(a1[m,0])


        glp = [(grad_W1,grad_b1),(grad_W2,grad_b2),(grad_W3,grad_b3)]
        return glp


    def vectorization_backpropagation_calculation(self, image, label, a_list, grad_l_param, l_param ,layers):
        W1 = l_param[0][0]
        W2 = l_param[1][0]
        W3 = l_param[2][0]

        grad_W1 = grad_l_param[0][0]
        grad_W2 = grad_l_param[1][0]
        grad_W3 = grad_l_param[2][0]

        grad_b1 = grad_l_param[0][1]
        grad_b2 = grad_l_param[1][1]
        grad_b3 = grad_l_param[2][1]

        a1 = a_list[0]
        a2 = a_list[1]
        a3 = a_list[2]

        # weight
        grad_W3 += (2 * (a3 - label) * a3 * (1 - a3)) @ np.transpose(a2)

            
        # bias
        grad_b3 += 2 * (a3 - label) * a3 * (1 - a3)
        
        # ---- 3rd layer
        # activation
        delta_3 = np.zeros((60, 1))
        delta_3 += np.transpose(W3) @ (2 *(a3 - label) * (a3 * (1 - a3)))
        # weight
        grad_W2 += (a2 * (1 - a2) * delta_3) @ np.transpose(a1)
        # bias
        grad_b2 += delta_3 * a2 * (1 - a2)

        # ---- 2nd layer
        # activation
        delta_2 = np.zeros((150, 1))

        delta_2 += np.transpose(W2) @ (delta_3 * a2 * (1 - a2))        
        # weight
        grad_W1 += (delta_2 * a1 * (1 - a1)) @ np.transpose(image)
        # bias
        grad_b1 += delta_2 * a1 * (1 - a1)

        glp = [(grad_W1,grad_b1),(grad_W2,grad_b2),(grad_W3,grad_b3)]
        
        return glp


    def train_network(self,
                        batch_size,
                        learning_rate,
                        number_of_epochs,
                        total_data ,
                        data_set,
                        layers_param,
                        layers,
                        vectorizaiton):

        total_costs = list()

        gradient_layers_parameters = list()

        for epoch in range(number_of_epochs):
            np.random.shuffle(data_set)
            # print(epoch + 1)
            batches = [data_set[x:x+batch_size] for x in range(0,200,batch_size)]
            for batch in batches:
                gradient_layers_parameters = ann.initialize_layers([102,150,60,4])
                a_list = list()
                for image,label in batch:
                    a_list = self.linear_forward_calculation(image,layers_param)

                    if vectorizaiton == True:
                        gradient_layers_parameters = self.vectorization_backpropagation_calculation(image,
                                                                                                    label,
                                                                                                    a_list,
                                                                                                    gradient_layers_parameters,
                                                                                                    layers_param,layers)
                    else:
                        gradient_layers_parameters = self.backpropagation_calculation(image,
                                                                                        label,
                                                                                        a_list,
                                                                                        gradient_layers_parameters,
                                                                                        layers_param,layers)
                lp = list()
                # update weights
                for p in range(0,len(layers_param)):
                    gw = gradient_layers_parameters[p][0]
                    gb = gradient_layers_parameters[p][1]
                    w = layers_param[p][0]
                    b = layers_param[p][1]

                    w = w - (learning_rate * (gw / batch_size))
                    b = b - (learning_rate * (gb / batch_size))

                    lp.append((w,b))
                layers_param = lp
            
            # calculate cost average per epoch
            cost = 0
            for data in data_set[:total_data]:
                al = self.linear_forward_calculation(data[0],layers_param)
                for j in range(4):
                    cost += np.power((al[-1][j, 0] - data[1][j,  0]), 2)
            cost /= 100
            total_costs.append(cost)  

        return total_costs,layers_param
        # print(a.shape)



    def accuracy(self, data_set, total_data, layers_parameters):
        a_list = list()
        correct_estimations = 0

        for data in data_set[:total_data]:
            a_list = self.linear_forward_calculation(data[0],layers_parameters)

            predicted_number = np.where(a_list[-1] == np.amax(a_list[-1]))
            real_number = np.where(data[1] == np.amax(data[1]))

            if predicted_number == real_number:
                correct_estimations += 1
        print("======================================================")
        print("TOTAL DATA: {}".format(total_data))
        print("------------------------------------------------------")
        print("CORRECT ESTIMATIONS: {}".format(correct_estimations))
        print("------------------------------------------------------")
        print("ACCURACY: {}".format((correct_estimations/total_data)*100))
        print("======================================================")
        return (correct_estimations/total_data)*100







if __name__ == "__main__":

    start_time = time.time()

    # ------------------------------
    # Part 1: Datasets
    # ------------------------------
    print("______________________ Datasets ______________________")

    ann = ANN_Network()
    

    # ------------------------------
    # Part 2: Feedforward
    # ------------------------------
    print("____________________ Feedforward _____________________")

    data_slice = 200
    layers = [102,150,60,4]
    layers_parameters = ann.initialize_layers(layers)
    ann.accuracy(ann.train_set,data_slice,layers_parameters)

    # Avarage feedforward - [expected accuracy: 25%]
    #                     - [expected time spent (each round): <5s]

    all_accuracys = list()
    num_of_round = 1000
    for i in range(num_of_round):
        layers_parameters = ann.initialize_layers(layers)
        acc = ann.accuracy(ann.train_set,data_slice,layers_parameters)
        all_accuracys.append(acc)
    print("======================================================")
    print("EXECUTION ROUNDS : {}".format(num_of_round))
    print("AVERAGE ACCURACY : {}".format(sum(all_accuracys)/num_of_round))
    print("======================================================")

    print("--- %s seconds ---" % (time.time() - start_time))
    
  
    # ------------------------------
    # Part 3: Backpropagation
    # ------------------------------
    print("___________________ Backpropagation __________________")

    start_time = time.time()

    data_slice = 200
    layers = [102,150,60,4]
    layers_parameters = ann.initialize_layers(layers)

    batch_size = 10
    learning_rate = 1
    number_of_epochs = 5
    vectorizaiton = False

    total_costs,layers_parameters = ann.train_network(batch_size,
                                                        learning_rate,
                                                        number_of_epochs,
                                                        data_slice,
                                                        ann.train_set,
                                                        layers_parameters,
                                                        layers,vectorizaiton)
    
    # Avarage backpropagation - [expected accuracy : <70%]
    #                         - [expected time spent (each round): <3m]
    #                         - [expected plot: descending]

    ann.accuracy(ann.train_set,data_slice,layers_parameters)
    print("--- %s seconds ---" % (time.time() - start_time))

    epoch_size = [x+1 for x in range(number_of_epochs)]
    plt.plot(epoch_size, total_costs)
    plt.show()


    # ------------------------------
    # Part 4: Vectorization
    # ------------------------------
    print("____________________ Vectorization ___________________")

    start_time = time.time()

    data_slice = 200
    layers = [102,150,60,4]
    layers_parameters = ann.initialize_layers(layers)

    batch_size = 10
    learning_rate = 1
    number_of_epochs = 20
    vectorizaiton = True

    total_costs,layers_parameters = ann.train_network(batch_size,
                                                        learning_rate,
                                                        number_of_epochs,
                                                        data_slice,
                                                        ann.train_set,
                                                        layers_parameters,
                                                        layers,vectorizaiton)
    
    # Avarage backpropagation - [expected accuracy: >90%]
    #                         - [expected time spent (each round): <10s]
    #                         - [expected plot: descending]

    ann.accuracy(ann.train_set,data_slice,layers_parameters)        
    print("--- %s seconds ---" % (time.time() - start_time))

    epoch_size = [x for x in range(number_of_epochs)]
    plt.plot(epoch_size, total_costs)
    plt.show()


    # # Run for 10 round    
    # Avarage backpropagation - [expected accuracy: >90%]
    #                         - [expected time spent (each round): <1m]
    #                         - [expected plot: increasing] 
    # all_accuracys = list()
    # for i in range(0,10):
    #     total_costs,layers_parameters = ann.train_network(batch_size,
    #                                                         learning_rate,
    #                                                         number_of_epochs,
    #                                                         data_slice,
    #                                                         ann.train_set,
    #                                                         layers_parameters,
    #                                                         layers,vectorizaiton)
    #     all_accuracys.append(ann.accuracy(ann.train_set,data_slice,layers_parameters))
        
    # print("--- %s seconds ---" % (time.time() - start_time))

    # execution_rounds = [x for x in range(10)]
    # plt.plot(execution_rounds, all_accuracys)
    # plt.show()



    # ------------------------------
    # Part 5: Testing 
    # ------------------------------
    print("_______________________ Testing ______________________")

    start_time = time.time()

    data_slice = len(ann.train_set)
    layers = [102,150,60,4]
    layers_parameters = ann.initialize_layers(layers)
    
    batch_size = 10
    learning_rate = 1
    number_of_epochs = 10
    vectorizaiton = True
    total_costs,layers_parameters = ann.train_network(batch_size,
                                                        learning_rate,
                                                        number_of_epochs,
                                                        data_slice,
                                                        ann.train_set,
                                                        layers_parameters,
                                                        layers,vectorizaiton)
    
    # Avarage backpropagation - [expected accuracy: <30%]
    #                         - [expected time spent (each round): <10s]
    #                         - [expected plot: descending]
    ann.accuracy(ann.test_set,data_slice,layers_parameters)
    print("--- %s seconds ---" % (time.time() - start_time))

    epoch_size = [x for x in range(number_of_epochs)]
    plt.plot(epoch_size, total_costs)
    plt.show()


    # # Run for 10 round
    # # Avarage backpropagation - [expected accuracy: >90%]
    # #                         - [expected time spent (each round): <1m]
    # #                         - [expected plot: increasing]   
    # all_accuracys = list()
    # for i in np.arange(0,10):
    #     total_costs,layers_parameters = ann.train_network(batch_size,
    #                                                         learning_rate,
    #                                                         number_of_epochs,
    #                                                         data_slice,
    #                                                         ann.train_set,
    #                                                         layers_parameters,
    #                                                         layers,vectorizaiton)
    #     all_accuracys.append(ann.accuracy(ann.train_set,data_slice,layers_parameters))

    # print(ann.accuracy(ann.test_set,data_slice,layers_parameters))
    # print("--- %s seconds ---" % (time.time() - start_time))

    # execution_rounds = [x for x in range(10)]
    # plt.plot(execution_rounds, all_accuracys)
    # plt.show()