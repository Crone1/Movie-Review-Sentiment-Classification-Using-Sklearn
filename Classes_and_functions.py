
# general packages
import os
import time
import numpy as np
import pandas as pd

# downlaod the data
from selenium import webdriver
import shutil

# read in the data
import tarfile

# for the Naive Bayes model
from sklearn.naive_bayes import MultinomialNB

# for evaluation
from sklearn import metrics


def get_download_folder():

    '''
    Find the Downloads folder on the users local PC

    Return:
        str - the path to the downloads folder
    '''

    home = os.path.expanduser("~")
    return os.path.join(home, "Downloads")


def load_data(data_directory, chromedriver_location):

    '''
    Load the documents in the data into a dictionary
    If the user has the data in the data directory specified then we use that
    If not, we downlaod it from the web and put it in this folder

    Params:
        data_directory: str - the path to the data directory that the user wants the data file to be in
        chromedriver_location - the location of the chromedriver on the users PC

    Return:
        dictionary - this maps a tuple to a list of lists
                   - (cross validation fold, document label): [[list of tokens in doc1], [list of tokens in doc2], ...]
    '''

    # get the path to where the file should be
    path_to_tar = os.path.abspath(os.path.join(data_directory, 'review_polarity.tar.gz'))

    # check if the our file doesn't exist in this directory
    if not os.path.isfile(path_to_tar):
        # if the file doesn't exist, we must download it and put it in the directory for next time
        data_url = 'https://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz'

        # Download the data
        driver = webdriver.Chrome(chromedriver_location)
        driver.get(data_url)
        time.sleep(5)
        driver.close()

        # Move the downloaded file to the apropriate location
        current_file_location = os.path.join(get_download_folder(), "review_polarity.tar.gz")
        shutil.move(current_file_location, path_to_tar)

    # we now know the file exists in our directory so can read the data from it
    counter = 0
    with open(path_to_tar, 'rb') as tgz_stream:
        data = {}
        with tarfile.open(mode='r|gz', fileobj=tgz_stream) as tar_archive:

            # iterate through the files in the tar folder
            for tar_member in tar_archive:

                # get the filepath components of the file
                path_components = tar_member.name.split('/')
                filename = path_components[-1]

                # exclude the README file
                if filename.startswith('cv') and filename.endswith('.txt') and '_' in filename:

                    # store this files data in a dictionary
                    label = path_components[-2]
                    fold = int(filename[2])
                    key = (fold, label)
                    if key not in data:
                        data[key] = []

                    # obtain the document and store it in the dictionary
                    f = tar_archive.extractfile(tar_member)
                    document = [line.decode('utf-8').split() for line in f.readlines()]
                    data[key].append(document)
                    counter += 1

    return data


def get_documents(data_dict, fold, label):

    '''
    Get the documents that fall under a specified fold and label pair

    Params:
        data_dict: dictionary - maps the cross validation fold & document label to the documents that fall under this fold-label
        fold: int - the cross validation fold that we want to get the documents for
        label: string - the label that we want to get the documents for (pos/neg)

    Return:
        list - a list of all the documents in the given dictionary that fall under the specified fold and label key
    '''

    return data_dict[(fold, label)]


def get_train_test_splits(data_dict):

    '''
    Use the folds used as keys in the dictionary to split the data into train and test sets with labels for cross-validation

    Params:
        data_dict: dictionary - maps the cross validation fold & document label to the documents that fall under this fold-label

    Returns:
        list: - a list of 'k' training and test set pairs
              - [(training_data_1, test_data_1), (training_data_2, test_data_2), ...]
    '''

    # get the folds in the dataset
    all_folds = set()
    all_labels = set()
    for (folds, label) in data_dict.keys():
        all_folds.add(folds)
        all_labels.add(label)

    # load the folds
    labelled_documents_in_folds = []
    for fold in all_folds:

        # get a list of all documents and their labels for this fold
        labelled_dataset = []
        for label in all_labels:
            for document in get_documents(data_dict, fold, label):
                labelled_dataset.append((document, label))

        # add these lists to a list for all the folds
        labelled_documents_in_folds.append(labelled_dataset)

    # create training-test splits from these labelled documents
    train_test_tuples = []
    for i in range(len(labelled_documents_in_folds)):
        test_data = labelled_documents_in_folds[i]
        training_data = []
        for j in range(len(labelled_documents_in_folds) - 1):

            fold_num = (i + j + 1) % len(labelled_documents_in_folds)
            assert fold_num != i
            training_data.extend(labelled_documents_in_folds[fold_num])

        train_test_tuples.append((training_data, test_data))

    return train_test_tuples


def get_document_preview(document, max_length):

    '''
    Preview a given number of characters from a given document

    Params:
        document: list of lists - a list of all tokens in a list of all sentences in a specified document
        max_length: int - the maximum number of characters to be output in our preview string

    Returns:
        string - a preview of the specified document that is less than 'max_length' characters long
    '''
    preview_words = []

    # iterate through the tokens in the document
    char_count = 0
    reached_limit = False
    for sentence in document:
        for token in sentence:

            # check if adding this character would create a string bigger than desired
            if char_count + len(token) + len(preview_words) > max_length:
                reached_limit = True
                break

            # add this token to the list of words in our preview
            preview_words.append(token)
            char_count += len(token)

        if reached_limit:
            break

    # create a string from the created list
    return '|'.join(preview_words)


# -----------------------------
# NAIVE BAYES WITH SCIKIT-LEARN
# -----------------------------

class Naive_Bayes:
    
    def __init__(self, clip_counts=True):
        self.vocab = set()
        self.token_to_vocab_index_dict = {}
        self.model = MultinomialNB()

        self.clip_counts = clip_counts


    def extract_document_features(self, data_with_labels):

        '''
        Create a matrix of features which corresponds to the documents X the words in the vocabulary
        The rows are the documents and the columns are the words
        We populate the matrix based on the occurances of the words in the documents

        Params:
            data_with_labels: list - this is a list of tuples containing a document and it's corresponging label

        Returns:
            numpy array - this is a matrix of features which corresponds to the documents X the words in the vocabulary
        '''

        # create numpy array of required size
        columns = len(self.vocab)
        rows = len(data_with_labels)
        feature_matrix = np.zeros((rows, columns), dtype=np.int32)

        # Populate feature matrix
        doc_index = 0
        for document, _ in data_with_labels:
            for sentence in document:
                for token in sentence:

                    # get the vocab index for this row
                    try:
                        vocab_index = self.token_to_vocab_index_dict[token]
                    except KeyError:
                        # token not in vocab --> skip this token & continue with next token
                        continue

                    # Populate the individual value in the feature matrix
                    if self.clip_counts:
                        feature_matrix[doc_index, vocab_index] = 1
                    else:
                        feature_matrix[doc_index, vocab_index] += 1
            doc_index += 1

        return feature_matrix


    def train(self, data_with_labels):
        
        '''
        Train the model on the given data

        Params:
            data_with_labels: list - this is a list of tuples containing a document and it's corresponging label

        Returns:
            None
        '''

        # define a set of all the vocabulary in the data using the input list of documents
        for document, _ in data_with_labels:
            for sentence in document:
                for token in sentence:
                    self.vocab.add(token)

        # create reverse map for fast token lookup
        for vocab_index, token in enumerate(self.vocab):
            self.token_to_vocab_index_dict[token] = vocab_index

        # extract the features - create numpy array of required size
        extracted_features = self.extract_document_features(data_with_labels)

        # create column vector with target labels - prepare target vector
        targets = get_targets(data_with_labels)

        # train the model on the features - pass numpy array to sklearn to train NB
        self.model.fit(extracted_features, targets)


    def predict(self, test_data):

        '''
        Take in some test documents and generate a predicted label for these documents using the trained model

        Params:
            test_data: list - a list of documents and their corresponding labels

        Returns:
            list - this is a list containing the models predicted labels for the given test set
        '''

        features = self.extract_document_features(test_data)

        # Get the predictions for model
        y_pred = self.model.predict(features)

        # Define the prediction labels
        labels = []
        for is_positive in y_pred:
            if is_positive:
                labels.append('pos')
            else:
                labels.append('neg')

        return labels


def get_targets(data_with_labels):

    '''
    Create a column vector of the target labels for all the documents in this dictionary of data

    Params:
        data_with_labels: list - this is a list of tuples containing a document and it's corresponging label

    Returns:
        numpy array - a one column array of 1's and 0's stating if a document is labelled as pos/neg
    '''

    # prepare target vector
    targets = np.zeros(len(data_with_labels), dtype=np.int8)

    # iterate through the list of documents & labels and populate the target vector
    index = 0
    for _, label in data_with_labels:
        if label == 'pos':
            targets[index] = 1
        index += 1

    return targets


def evaluate_predictions(test_data, y_pred):

    '''
    Evaluate the predicted labels for a given test set of data using the actual labels for this set

    Params:
        test_data: list - a list of documents and their corresponding labels
        y_pred: list - this is a list containing the models predicted labels for the given test set

    Returns:
        tuple - (an accuracy score for the predictions, a confusion matrix evaluating the predictions)
    '''

    # turn the predicted variables into a binary 1 or 0 based on 'pos' or 'neg'
    binary_y_pred = np.where(np.array(y_pred)=='pos', 1, 0)

    # generate the actual y values for this test data
    y_true = get_targets(test_data)

    # get an accuracy score for these predictions and also get a confusion matrix
    return metrics.accuracy_score(y_true, binary_y_pred), metrics.confusion_matrix(y_true, binary_y_pred)


def print_first_n_predictions(model, test_data, num_predictions, len_preview):

    '''
    Use a model to generate a specified number of predictions and output a table containing the predicted label and the actual abel

    Params:
        model: Class - a class that defines a model
        test_data: list - a list of documents and their corresponding labels
        num_predictions: int - the number of prediction Vs actual rows we want in our output dataframe
        len_preview: int - the maximum number of characters to be output in our preview string

    Returns:
        pandas dataframe - a dataframe containing the predicted label side by side with the actual label for a given document
    '''

    # get the models predictions for the test data
    predictions = model.predict(test_data)

    # evaluate the generated predictions and output the results
    accuracy, confusion_matrix = evaluate_predictions(test_data, predictions)
    print("Accuracy =", accuracy)
    print("Confusion_matrix:\n", confusion_matrix)

    # create a dataframe displaying a specified number of these predictions
    pred_df = pd.DataFrame(columns=['label', 'prediction', 'documents'])
    for i in range(num_predictions):
        document, label = test_data[i]
        doc_preview = get_document_preview(document, len_preview)

        # add this prediction to the dataframe of all output predictions
        one_pred = pd.DataFrame({'label':label, 'prediction':predictions[i], 'documents':doc_preview}, index=[0])
        pred_df = pd.concat([pred_df, one_pred], axis=0)

    return pred_df.reset_index(drop=True)


def evaluate_model(model, train_test_tuples, verbose=False):

    '''
    Evalute the model by using cross validation to train the model on the different training sets and
    generating a prediction on the test sets
    
    Params:
        model: Class - a class that defines a model
        train_test_tuples: list: - a list of 'k' training and test set tuples - eg. (training_data_1, test_data_1)
        verbose: boolean - whether we want to print the model accuracy scores as we go along in trining the model on them

    Returns:
        tuple of ints: (the average model accuracy, the max accuracy, the min accuracy) across all train-test sets we trained on
    '''

    # iterate through the cross validation train-test sets
    accuracies = []
    fold = 1
    for train_data, test_data in train_test_tuples:

        # train the model on each set as we iterate
        model.train(train_data)

        # get an accuracy score for the trained model
        predictions = model.predict(test_data)
        accuracy, _ = evaluate_predictions(test_data, predictions)
        accuracies.append(accuracy)

        # output the model accuracy if requested
        if verbose:
            print('Fold {} of {} --> {}'.format(fold, len(train_test_tuples), accuracy))
            fold += 1

    # create the evalutation summary statistics for this model
    n = float(len(accuracies))
    avg = sum(accuracies) / n
    mse = sum([(x-avg)**2 for x in accuracies]) / n

    return avg, mse**0.5, min(accuracies), max(accuracies)