#!/usr/bin/env python
# coding: utf-8

# In[1]:
# John Andray William Doney / 2201812570
# Derren Delano Soerjadi / 2201740571

import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


# In[2]:



def loadData():

    data = pd.read_csv("E202-COMP7117-TD01-00 - classification.csv")

    if data.isna().values.any() == True:
        data = data.dropna()
    
    dataInput = data[["volatile acidity", "chlorides", "free sulfur dioxide", "total sulfur dioxide", 
                      "density", "pH","sulphates","alcohol"]]
    target = data[["quality"]]
    
    # print(dataInput.shape)
    
    for y in dataInput:
        if(y == "free sulfur dioxide" ):
#             print(y)
#             print(dataInput[y][0])
            for j in range(len(dataInput)):
                if(dataInput[y][j] == "High"):
                    dataInput.at[j, y] = 3
                elif(dataInput[y][j] == "Medium"):
                    dataInput.at[j, y] = 2
                elif(dataInput[y][j] == "Low"):
                    dataInput.at[j, y] = 1
                else:
                    dataInput.at[j, y] = 0
        elif(y == "density"):
#             print(y)
#             print(dataInput[y][0])
            for j in range(len(dataInput)):
                if(dataInput[y][j] == "Very High"):
                    dataInput.at[j, y] = 0
                elif(dataInput[y][j] == "High"):
                    dataInput.at[j, y] = 3
                elif(dataInput[y][j] == "Medium"):
                    dataInput.at[j, y] = 2
                elif(dataInput[y][j] == "Low"):
                    dataInput.at[j, y] = 1
        elif(y == "pH"):
#             print(y)
#             print(dataInput[[y]].values[0])
            for j in range(len(dataInput)):
                    if(dataInput[y][j] == "Very Basic"):
                        dataInput.at[j, y] = 3
                    elif(dataInput[y][j] == "Normal"):
                        dataInput.at[j, y] = 2
                    elif(dataInput[y][j] == "Very Acidic"):
                        dataInput.at[j, y] = 1
                    else:
                        dataInput.at[j, y] = 0
    
    #normalisasi
    datainput = MinMaxScaler().fit_transform(dataInput)
    target = OneHotEncoder(sparse=False).fit_transform(target)
    
    #PCA
    datainput = PCA(n_components=4).fit_transform(dataInput)
        
    return datainput, target


inputData, target = loadData()
# print(inputData)


# In[3]:



layers = {
    "input": 4, #8 different kind of data
    "hidden": 500,
    "output": 5 # decent, fair, fine, good, great
}

weights = {
    'input_to_hidden' : tf.Variable(tf.random_normal([layers['input'], layers['hidden']])),
    'hidden_to_output' : tf.Variable(tf.random_normal([layers['hidden'], layers['output']]))
}

bias = {
    'input_to_hidden' : tf.Variable(tf.random_normal([layers['hidden']])),
    'hidden_to_output' : tf.Variable(tf.random_normal([layers['output']]))
}

inputPlaceholder = tf.placeholder(tf.float32, [None, layers["input"]])
outputPlacehlder = tf.placeholder(tf.float32, [None, layers["output"]])


# In[4]:



def feedForward(inputData):
    #first
    input_to_hidden_bias = tf.matmul(inputData, weights['input_to_hidden']) + bias['input_to_hidden']
    activated_input_to_hidden = tf.nn.sigmoid(input_to_hidden_bias)
    #second
    hidden_to_output_bias = tf.matmul(activated_input_to_hidden, weights['hidden_to_output']) + bias['hidden_to_output']
    activated_hidden_to_output = tf.nn.sigmoid(hidden_to_output_bias)

    return activated_hidden_to_output

predict = feedForward(inputPlaceholder)

epoch = 5000

error = tf.reduce_mean(0.5 * ( outputPlacehlder - predict ) ** 2)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(error)

inputTrain, inputTest, outputTrain, outputTest = train_test_split(inputData, target, test_size=0.1)
# print(inputData.shape)
# print(target.shape)
inputTrain, inputValidationTest, outputTrain, outputValidationTest = train_test_split(inputTrain, outputTrain, test_size=0.2)

# print(inputTrain.shape)
# print(inputValidationTest.shape)
# print(inputTest.shape)


# In[5]:



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #70% of the dataset - train
    for i in range(1, epoch + 1) :
        train_dict = {
            inputPlaceholder : inputTrain,
            outputPlacehlder : outputTrain
        }
        sess.run(train, feed_dict = train_dict)

        loss = sess.run(error, feed_dict = train_dict)

        if i % 100 == 0:
            print("Epoch : {}, loss : {}".format(i, loss))
        
        #20% of the dataset - valid
        if i % 500 == 0:
            
            validation_dict = {
                inputPlaceholder : inputValidationTest,
                outputPlacehlder : outputValidationTest
            }
            sess.run(train, feed_dict = validation_dict)

            Validationloss = sess.run(error, feed_dict = validation_dict)
            
#             print("Validation Epoch : {}, loss : {}".format(i, Validationloss))
            if i == 500:
                lowestValidationLoss = Validationloss
            
                f=open("lowestValidationLoss.txt", "w")
                f.write(str(lowestValidationLoss))
                f.close()
            
            if Validationloss < lowestValidationLoss:
                lowestValidationLoss = Validationloss
                f=open("lowestValidationLoss.txt", "w")
                f.write(str(lowestValidationLoss))
                f.close()
            
                
    #10% of the dataset - evaluation
    matches = tf.equal(tf.argmax(outputPlacehlder,axis = 1), tf.argmax(predict,axis = 1))
    accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))

    feed_test = {
        inputPlaceholder: inputTest,
        outputPlacehlder: outputTest
    }

    print("accuracy: {}".format(sess.run(accuracy, feed_dict = feed_test) *100 ))
    


# In[ ]:





# In[ ]:





# In[ ]:




