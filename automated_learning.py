import graphviz
import pandas as pd
from tabulate import tabulate
from sklearn import tree, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Get all the data
astra_data = pd.read_csv('asteroid_impacts.csv', index_col = 0)

# Assign data from first 14 columns to X variable
x = astra_data.iloc[:, 0:14]

# Assign data from the final column to y variable
y = astra_data.iloc[:, 14:15]

# Transform data
le = preprocessing.LabelEncoder()
x = x.apply(le.fit_transform)
y = y.apply(le.fit_transform)

# Get column names
names = x.columns.values

# Split train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)

# Scale data
scaler = preprocessing.StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

'''
Variables
x_train is all the data to train the model wihout the final answer
y_train is the answer of the data for training
x_test is all the data to test the model wihout the final answer
y_train is the answer of the data for testing
'''


# Neural Network
def neural_network(x_train, y_train, x_test, y_test):

    # Testing different parameters 
    data = []
    test_1 = {'mlp': MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000),
                'parameters': 'hidden_layer_sizes=(10, 10, 10), max_iter=1000',
                'test': 'Test 1'}
    data.append(test_1)

    test_2 = {'mlp': MLPClassifier(hidden_layer_sizes=(10, 20, 10), max_iter=1000),
                'parameters': 'hidden_layer_sizes=(10, 20, 10), max_iter=1000',
                'test': 'Test 2'}
    data.append(test_2)

    test_3 = {'mlp': MLPClassifier(hidden_layer_sizes=(10, 10, 10, 10), max_iter=10000),
                'parameters': 'hidden_layer_sizes=(10, 10, 10, 10), max_iter=10000',
                'test': 'Test 3'}
    data.append(test_3)

    test_4 = {'mlp': MLPClassifier(learning_rate_init=0.5, max_iter=1000),
                'parameters': 'learning_rate_init=0.5, max_iter=1000',
                'test': 'Test 4'}
    data.append(test_4)

    test_5 = {'mlp': MLPClassifier(learning_rate_init=0.05, max_iter=1000),
                'parameters': 'learning_rate_init=0.05, max_iter=1000',
                'test': 'Test 5'}
    data.append(test_5)

    test_6 = {'mlp': MLPClassifier(max_iter=1000),
                'parameters': 'max_iter=1000',
                'test': 'Test 6'}
    data.append(test_6)

    for d in data:

        # Train neural network
        d['mlp'] = d['mlp'].fit(x_train, y_train.values.ravel())

        # Predict
        predictions = d['mlp'].predict(x_test)
        d['predictions'] = predictions

        # Calculate accuracy
        percentage = 100 * sum([1 for i in range(len(predictions)) if predictions[i] == y_test.values[i]])/len(predictions)
        d['accuracy'] = percentage

    # Sort all options by accuracy
    data = sorted(data, key=lambda k: k['accuracy'], reverse=True)
    table_data = [{'Test':x['test'], 'Parameters':x['parameters'], 'Accuracy':"{:.4f}".format(x['accuracy']) + ' %'} for x in data]

    # Get the best option
    best_option = data[0]

    # Results
    print('\n##################\n# Neural Network #\n##################')
    print('\nAll different tests')
    print(tabulate(table_data, headers='keys', tablefmt='fancy_grid', showindex=True))
    # Final answere
    print('\n------------------------------------')
    print('Best Test: ', best_option['test'])
    print('Parameters Used: ', best_option['parameters'])
    print('Accuracy: ', best_option['accuracy'], '%')
    print('------------------------------------')
    print('\nConfusion Matrix')
    matrix = confusion_matrix(y_test, best_option['predictions'])
    matrix_data = {
                    '': ['No Hazard', 'Hazard'], 
                    'Correct': [matrix[0][0], matrix[1][1]], 
                    'Incorrect': [matrix[0][1], matrix[1][0]]
                }
    print(tabulate(matrix_data, headers='keys', tablefmt='fancy_grid'))
    print('\nNeural Network Final Report')
    print(classification_report(y_test, best_option['predictions']))

    return best_option


# Decision Tree
def decision_tree(names, x_train, y_train, x_test, y_test):

    # Testing different parameters 
    data = []
    test_1 = {'clf': tree.DecisionTreeClassifier(criterion = 'entropy'),
                'parameters': 'criterion=entropy',
                'test': 'Test 1'}
    data.append(test_1)

    test_2 = {'clf': tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 1),
                'parameters': 'criterion=entropy, max_depth=1',
                'test': 'Test 2'}
    data.append(test_2)

    test_3 = {'clf': tree.DecisionTreeClassifier(criterion = 'entropy', splitter = 'random'),
                'parameters': 'criterion=entropy, splitter=random',
                'test': 'Test 3'}
    data.append(test_3)

    test_4 = {'clf': tree.DecisionTreeClassifier(criterion = 'entropy', splitter = 'random', max_depth = 2),
                'parameters': 'criterion=entropy, splitter=random, max_depth=2',
                'test': 'Test 4'}
    data.append(test_4)

    test_5 = {'clf': tree.DecisionTreeClassifier(criterion = 'gini', splitter = 'random'),
                'parameters': 'criterion=gini, splitter=random',
                'test': 'Test 5'}
    data.append(test_5)

    test_6 = {'clf': tree.DecisionTreeClassifier(criterion = 'gini', splitter = 'random', max_depth = 3),
                'parameters': 'criterion=gini, splitter=random, max_depth=3',
                'test': 'Test 6'}
    data.append(test_6)

    for d in data:

        # Create the model
        d['clf'] = d['clf'].fit(x_train, y_train)

        # Create graphic results
        dot_data = tree.export_graphviz(d['clf'], out_file = None,
                                        feature_names = list(names),
                                        class_names = ["Hazard", "No Hazard"],
                                        filled = False, rounded = True,
                                        special_characters = True)

        # Save graphic
        d['graph'] = graphviz.Source(dot_data)

        # Predict test answers
        predictions = d['clf'].predict(x_test)
        d['predictions'] = predictions

        # Calculate accuracy
        percentage = 100 * sum([1 for i in range(len(predictions)) if predictions[i] == y_test.values[i]])/len(predictions)
        d['accuracy'] = percentage

    # Sort all options by accuracy
    data = sorted(data, key=lambda k: k['accuracy'], reverse=True)
    table_data = [{'Test':x['test'], 'Parameters':x['parameters'], 'Accuracy':"{:.4f}".format(x['accuracy']) + ' %'} for x in data]

    # Get the best option
    best_option = data[0]

    # Print results as a table
    print('\n#################\n# Decision Tree #\n#################')
    print('\nAll different tests')
    print(tabulate(table_data, headers='keys', tablefmt='fancy_grid', showindex=True))
    # Final answere
    print('\n------------------------------------')
    print('Best Test: ', best_option['test'])
    print('Parameters Used: ', best_option['parameters'])
    print('Accuracy: ', best_option['accuracy'], '%')
    print('------------------------------------')
    print('\nConfusion Matrix')
    matrix = confusion_matrix(y_test, best_option['predictions'])
    # Create dictionary of the confusion matrix to print it as table
    matrix_data = {
                    '': ['No Hazard', 'Hazard'], 
                    'Correct': [matrix[0][0], matrix[1][1]], 
                    'Incorrect': [matrix[0][1], matrix[1][0]]
                }
    print(tabulate(matrix_data, headers='keys', tablefmt='fancy_grid'))
    print('\nDecision Tree Final Report')
    print(classification_report(y_test, best_option['predictions']))
    # Save the tree as png file
    best_option['graph'].render('tree', view=True, format='png')

    return best_option


def results():

    # Get the results
    neural_results = neural_network(x_train, y_train, x_test, y_test)
    decision_tree_results = decision_tree(names, x_train, y_train, x_test, y_test)

    # Create dictionary for final report
    final_report = [
                        {
                            'Algorithm': 'Neural Network', 
                            'Parameters': neural_results['parameters'], 
                            'Accuracy': neural_results['accuracy']
                        },
                        {
                            'Algorithm': 'Decision Tree', 
                            'Parameters': decision_tree_results['parameters'], 
                            'Accuracy': decision_tree_results['accuracy']
                        }
                    ]

    # Sort by accuray
    final_report = sorted(final_report, key=lambda k: k['Accuracy'], reverse=True)

    # Format as percentage accuracy
    for f in final_report:
        f['Accuracy'] = '{:.4f}'.format(f['Accuracy']) + ' %'
    
    # Print final report as table
    print('\nFinal Report')
    print(tabulate(final_report, headers='keys', tablefmt='fancy_grid'))
    print('The best algorithm for this prediction is:', final_report[0]['Algorithm'])

results()
