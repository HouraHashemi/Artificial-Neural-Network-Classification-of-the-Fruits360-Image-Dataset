import numpy as np
import random
import pickle

class Loading_Datasets:


    def __init__(self, train_plk_path, test_plk_path):
      
        self.train_set = self.load_train_set_features(train_plk_path)
        self.test_set = self.load_test_set_features(test_plk_path)


    def load_train_set_features(self, path):
        # loading training set features
        f = open("{}/Datasets/train_set_features.pkl".format(path), "rb")
        train_set_features2 = pickle.load(f)
        f.close()

        # reducing feature vector length 
        features_STDs = np.std(a=train_set_features2, axis=0)
        train_set_features = train_set_features2[:, features_STDs > 52.3]

        # changing the range of data between 0 and 1
        train_set_features = np.divide(train_set_features, train_set_features.max())

        # loading training set labels
        f = open("{}/Datasets/train_set_labels.pkl".format(path), "rb")
        train_set_labels = pickle.load(f)
        f.close()

        # ------------
        # preparing our training and test sets - joining datasets and lables
        train_set = []
        for i in range(len(train_set_features)):
            label = np.array([0,0,0,0])
            label[int(train_set_labels[i])] = 1
            label = label.reshape(4,1)
            train_set.append((train_set_features[i].reshape(102,1), label))
        # shuffle
        random.shuffle(train_set)
        train_set = np.array(train_set,dtype=object)
        
        # print(len(train_set)) #1962
        return train_set


    def load_test_set_features(self, path):
        # ------------
        # loading test set features
        f = open("{}/Datasets/test_set_features.pkl".format(path), "rb")
        test_set_features2 = pickle.load(f)
        f.close()

        # reducing feature vector length 
        features_STDs = np.std(a=test_set_features2, axis=0)
        test_set_features = test_set_features2[:, features_STDs > 48]

        # changing the range of data between 0 and 1
        test_set_features = np.divide(test_set_features, test_set_features.max())

        # loading test set labels
        f = open("{}/Datasets/test_set_labels.pkl".format(path), "rb")
        test_set_labels = pickle.load(f)
        f.close()


        # ------------
        # preparing our training and test sets - joining datasets and lables
        test_set = []
        for i in range(len(test_set_features)):
            label = np.array([0,0,0,0])
            label[int(test_set_labels[i])] = 1
            label = label.reshape(4,1)
            test_set.append((test_set_features[i].reshape(102,1), label))
        # shuffle
        random.shuffle(test_set)
        test_set = np.array(test_set,dtype=object)
        # print(len(test_set)) #662
        return test_set


