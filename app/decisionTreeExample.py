# https://scikit-learn.org/stable/modules/tree.html

from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def prepare_training_data():
    # create a list, the list will have all the possible combinations of alive, #of neighbors
    # this is for one cell: [is_cell_alive_now, number_of_neighbours]
    input_combos_list = [
        [False, 0],
        [False, 1],
        [False, 2],
        [False, 3],
        [False, 4],
        [False, 5],
        [False, 6],
        [False, 7],
        [False, 8],
        [True, 0],
        [True, 1],
        [True, 2],
        [True, 3],
        [True, 4],
        [True, 5],
        [True, 6],
        [True, 7],
        [True, 8]
    ]
    # 
    expected_outputs = [
    # Dead Cells
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
    # Live Cells
        False,
        False,
        True,
        True,
        False,
        False,
        False,
        False,
        False
    ]
    train_test_split(input_combos_list, expected_outputs, test_size=0.33, random_state=42)

    return input_combos_list, expected_outputs


def game_of_life():
    trained_classifier = tree.DecisionTreeClassifier()
    trained_classifier = trained_classifier.fit(prepare_training_data())


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