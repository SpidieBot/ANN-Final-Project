#!/usr/bin/env python
# coding: utf-8

# In[1]:
# John Andray William Doney / 2201812570
# Derren Delano Soerjadi / 2201740571

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.decomposition import PCA


# In[2]:


class SOM:
    def __init__(self, width, height, input_dimension):
        self.width = width
        self.height = height
        self.input_dimension = input_dimension

        self.weight = tf.Variable(tf.random_normal([width * height, input_dimension]))
        self.input = tf.placeholder(tf.float32, [input_dimension])

        self.location = [tf.to_float([y,x]) for y in range(height) for x in range(width)]

        self.bmu = self.getBMU()

        self.update_weight = self.update_neigbours()

    def getBMU(self):
        #Best Matching Unit

        #Eucledian distance
        square_distance = tf.square(self.input - self.weight)
        distance = tf.sqrt(tf.reduce_sum(square_distance, axis=1))

        #Get BMU index
        bmu_index = tf.argmin(distance)
        #Get the position
        bmu_position = tf.to_float([tf.div(bmu_index,self.width), tf.mod(bmu_index, self.width)])
        return bmu_position

    def update_neigbours(self):

        learning_rate = 0.1

        #Formula calculate sigma / radius
        sigma = tf.to_float(tf.maximum(self.width, self.height) / 2)

        #Eucledian Distance between BMU and location
        square_difference = tf.square(self.bmu - self.location)
        distance = tf.sqrt(tf.reduce_sum(square_difference,axis=1))

        #Calculate Neighbour Strength based on formula
        # NS = tf.exp((- distance ** 2) /  (2 * sigma ** 2))
        NS = tf.exp(tf.div(tf.negative(tf.square(distance)), 2 * tf.square(sigma)))

        #Calculate rate before reshape
        rate = NS * learning_rate

        #Reshape to [width * height, input_dimension]
        rate_stacked = tf.stack([tf.tile(tf.slice(rate,[i],[1]), [self.input_dimension]) 
            for i in range(self.width * self.height)])

        #Calculate New Weight
        new_weight = self.weight + rate_stacked * (self.input - self.weight)

        return tf.assign(self.weight, new_weight)

    def train(self, dataset, epoch):

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            #training
            for i in range(epoch+1):
                for data in dataset:
                    dictionary = {
                        self.input : data
                    }

                    sess.run(self.update_weight,feed_dict=dictionary)

            #assign clusters
            location = sess.run(self.location)
            weight = sess.run(self.weight)

            clusters = [[] for i in range(self.height)]

            for i, loc in enumerate(location):
                clusters[int(loc[0])].append(weight[i])

            self.clusters = clusters





# In[3]:



def loadData():
    
    data = pd.read_csv("E202-COMP7117-TD01-00 - clustering.csv")

    if data.isna().values.any() == True:
        data = data.dropna()
    
    dataInput = data[["ProductRelated_Duration","ExitRates","SpecialDay","VisitorType","Weekend"]]
    
    for y in dataInput:
        if(y == "SpecialDay" ):
#             print(y)
#             print(dataInput[y][0])
            for j in range(len(dataInput)):
                if(dataInput[y][j] == "HIGH"):
                    dataInput.at[j, y] = 2
                elif(dataInput[y][j] == "NORMAL"):
                    dataInput.at[j, y] = 1
                elif(dataInput[y][j] == "LOW"):
                    dataInput.at[j, y] = 0
        elif(y == "VisitorType" ):
#             print(y)
#             print(dataInput[y][0])
            for j in range(len(dataInput)):
                if(dataInput[y][j] == "Returning_Visitor"):
                    dataInput.at[j, y] = 2
                elif(dataInput[y][j] == "New_Visitor"):
                    dataInput.at[j, y] = 1
                elif(dataInput[y][j] == "Other"):
                    dataInput.at[j, y] = 0
        elif(y == "Weekend" ):
#             print(y)
#             print(dataInput[[y]].values[0])
            dataInput[y] = dataInput[y].astype(int)
                    
    # print(dataInput)
    # print(dataInput.dtypes)
    #normalisasi
    
    dataInput = MinMaxScaler().fit_transform(dataInput)
    dataInput = PCA(n_components=3).fit_transform(dataInput)

    return dataInput


colors_dataset = loadData()
# print(colors_dataset.shape)
# print(colors_dataset)


# In[4]:



input_dimension = len(colors_dataset[0])
# print(input_dimension)
epoch = 5000


# In[5]:



som = SOM(15,15,input_dimension)

som.train(colors_dataset,epoch)
plt.imshow(som.clusters)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




