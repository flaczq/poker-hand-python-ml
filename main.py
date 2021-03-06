import numpy as np
import random
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def translate_features(num_features):
    translated = ''
    for i, num_feature in enumerate(num_features):
        if i % 2 == 0:
            translated += translate_suit(num_feature)
        else:
            translated += translate_rank(num_feature) + ' '
    return translated


def translate_suit(num_suit):
    # 1,3,5,7,9: 1 - HEART, 2 - SPADES, 3 - DIAMONDS, 4 - CLUBS
    if num_suit == 1:
        return '(H)'
    elif num_suit == 2:
        return '(S)'
    elif num_suit == 3:
        return '(D)'
    elif num_suit == 4:
        return '(C)'
    return num_suit


def translate_rank(num_rank):
    # 2,4,6,8,10: 1 - ACE, 2 - 10, 11 - JACK, 12 - QUEEN, 13 - KING
    if num_rank == 1:
        return 'A'
    elif num_rank >= 2 and num_rank <= 10:
        return str(num_rank)
    elif num_rank == 11:
        return 'J'
    elif num_rank == 12:
        return 'Q'
    elif num_rank == 13:
        return 'K'
    return num_rank


def translate_class(num_class):
    # 11: 0 - NOTHING, 1 - ONE PAIR, 2 - TWO PAIRS, 3 - THREE OF KIND, 4 - STRAIGHT, 5 - FLUSH, 6 - FULL HOUSE, 7 - FOUR OF KIND, 8 - STRAIGHT FLUSH, 9 - ROYAL FLUSH
    if num_class == 0:
        return 'NOTHING'
    elif num_class == 1:
        return 'ONE PAIR'
    elif num_class == 2:
        return 'TWO PAIRS'
    elif num_class == 3:
        return 'THREE OF KIND'
    elif num_class == 4:
        return 'STRAIGHT'
    elif num_class == 5:
        return 'FLUSH'
    elif num_class == 6:
        return 'FULL HOUSE'
    elif num_class == 7:
        return 'FOUR OF KIND'
    elif num_class == 8:
        return 'STRAIGHT FLUSH'
    elif num_class == 9:
        return 'ROYAL FLUSH'
    return num_class


def start():
    print('-------- BEGIN --------')
    print('----- POKER HANDS -----')
    print('--  MACHINE LEARNING --\n')


def stop():
    print('\n--  MACHINE LEARNING --')
    print('----- POKER HANDS -----')
    print('--------- END ---------')


def main():
    '''
    1)  Card #1 suit
    2)  Card #1 rank
    3)  Card #2 suit
    4)  Card #2 rank
    5)  Card #3 suit
    6)  Card #3 rank
    7)  Card #4 suit
    8)  Card #4 rank
    9)  Card #5 suit
    10) Card #5 rank
    11) Hand class
    '''
    # PREPARING
    print('Preparing...')
    training_filename = 'poker-hand-training-true.data'
    training_file = open(training_filename)
    training_data = np.loadtxt(training_file, delimiter=',', dtype='float')
    training_features = training_data[:, :-1]
    training_labels = training_data[:, -1]

    testing_filename = 'poker-hand-testing.data'
    testing_file = open(testing_filename)
    testing_data = np.loadtxt(testing_file, delimiter=',', dtype='float')
    testing_features = testing_data[:, :-1]
    testing_labels = testing_data[:, -1]

    # SCALING
    # print('Scaling...')
    # scaler = StandardScaler()
    # scaler.fit(training_features)
    # training_features = scaler.transform(training_features)
    # testing_features = scaler.transform(testing_features)

    # TRAINING
    print('Training...')
    names = [
        # 'KNeighborsClassifier(3)',
        # 'SVC(gamma=2, C=1)',
        # 'GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)',
        # 'DecisionTreeClassifier(max_depth=5)',
        # 'RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)',
        # 'MLPClassifier(alpha=1)',
        'AdaBoostClassifier',
        # 'GaussianNB',
        # 'QuadraticDiscriminantAnalysis'
    ]
    classifiers = [
        # KNeighborsClassifier(3),
        # SVC(gamma=2, C=1),
        # GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
        # DecisionTreeClassifier(max_depth=5),
        # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        # MLPClassifier(alpha=1),
        AdaBoostClassifier(),
        # GaussianNB(),
        # QuadraticDiscriminantAnalysis()
    ]
    for clsIndex in range(len(classifiers)):
        classifier = classifiers[clsIndex].fit(
            training_features, training_labels)

        # TESTING
        print(f'{names[clsIndex]}: Testing...\n')
        prediction = classifier.predict(testing_features)
        prediction_proba = classifier.predict_proba(testing_features)

        # SHOWING 10 RANDOM PREDICTIONS
        for i in random.sample(range(len(testing_features)), 10):
            correct = prediction[i] == testing_labels[i]
            print(
                f'#{i}: {translate_features(testing_features[i])} -> {translate_class(prediction[i])} [{correct}]')

        print('Accuracy: {:.4%}'.format(
            accuracy_score(testing_labels, prediction)))
        print('Log loss: {}'.format(log_loss(testing_labels, prediction_proba)))


start()
main()
stop()
