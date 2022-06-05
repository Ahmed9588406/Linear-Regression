import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
predict = "G3"
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
# i created a for loop in my model to get the nearest result of my model

best = 0
"""for _ in range(30):
     x_train , x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

     #creating a training model
     Linear =linear_model.LinearRegression()

#using x_train data and y_train data finding the best fit line among them
Linear.fit(x_train,y_train )

#using to score x_test and y_test
#that Will return a value accurce of our model
acc = Linear.score(x_test, y_test)
print(acc)
if acc > best:
    best =acc
    with open("studentmodel.pickel", "wb")as f:
     #we use dump to save our model
      pickle.dump(Linear, f)"""
pickle_in = open("studentmodel.pickel", "rb")
# that will load our model in to the variable linear
Linear = pickle.load(pickle_in)

# the cooficient from the equation y=mx+b and cooficents stands for m give us the list of different cooficient
# give us the cooficient of 5 different variables i used in line 9
print('coofficiont: \n', Linear.coef_)

# show us the y intercept
print('intercept: \n', Linear.intercept_)

prediction = Linear.predict(x_test)
# x_test stands for the predicted result of the 5 variables in data and y_test stands for the main value of the data in the file we read in this project
for x in range(len(prediction)):
    print(prediction[x], x_test[x], y_test[x])
# i used style.use and pyplot.scatter to plot my model on the screen
# I Created this variable to be one of the attributes
p = 'G1'
style.use("ggplot")
# this needs a X value = p = 'G1' and Y value =data["G3"] on the ploting graph
pyplot.scatter(data[p], data["G3"])
# making label for the axis
# after you see the graph when you run the program you will see the corelation between the data in x_axis and y_axis
pyplot.xlabel(p)
pyplot.ylabel("Final Grades")
pyplot.show()


