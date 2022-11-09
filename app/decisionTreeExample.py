# https://scikit-learn.org/stable/modules/tree.html

from sklearn import tree
from sklearn.datasets import load_iris


def example1():
    # You may hard code your data as given or to use a .csv file import csv then fetch your data from .csv file

    # Assume we have two dimensional feature space with two classes we like distinguish
    dataTable = [[2,9],[4,10],[5,7],[8,3],[9,1]]

    dataLabels = ["Class A","Class A","Class B","Class B","Class B"]

    # Declare our classifier
    trained_classifier = tree.DecisionTreeClassifier()

    # Train our classifier with data we have
    trained_classifier = trained_classifier.fit(dataTable,dataLabels)

    # We are done with training, so it is time to test it!
    someDataOutOfTrainingSet = [[10,2]]
    label = trained_classifier.predict(someDataOutOfTrainingSet)

    # Show the prediction of trained classifier for data [10,2]
    print(label[0])

    tree.export_graphviz(trained_classifier, "img/tree.dot")
    # dot -Tpng tree.dot -o tree.png


def example2():
    
    iris = load_iris()
    x, y = iris.data, iris.target
    print(x)
    print("------")
    print(y)
    print("------")
    print(iris.feature_names)
    print("------")
    print(iris.target_names)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x, y)

    tree.export_graphviz(clf, out_file="img/tree2.dot", 
        feature_names=iris.feature_names,  
        class_names=iris.target_names,  
        filled=True, rounded=True,  
        special_characters=True)
    # dot -Tpng tree2.dot -o tree2.png

# example1()
example2()