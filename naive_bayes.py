import numpy as np
import os


class NaiveBayes():
    def __init__(self):
        self.num_train_hams = 0
        self.num_train_spams = 0
        self.word_counts_spam = {}
        self.word_counts_ham = {}
        self.HAM_LABEL = 'ham'
        self.SPAM_LABEL = 'spam'

    def load_data(self, path:str='data/'):
        assert set(os.listdir(path)) == set(['test', 'train'])
        assert set(os.listdir(os.path.join(path, 'test'))) == set(['ham', 'spam'])
        assert set(os.listdir(os.path.join(path, 'train'))) == set(['ham', 'spam'])

        train_hams, train_spams, test_hams, test_spams = [], [], [], []
        for filename in os.listdir(os.path.join(path, 'train', 'ham')):
            train_hams.append(os.path.join(path, 'train', 'ham', filename))
        for filename in os.listdir(os.path.join(path, 'train', 'spam')):
            train_spams.append(os.path.join(path, 'train', 'spam', filename))
        for filename in os.listdir(os.path.join(path, 'test', 'ham')):
            test_hams.append(os.path.join(path, 'test', 'ham', filename))
        for filename in os.listdir(os.path.join(path, 'test', 'spam')):
            test_spams.append(os.path.join(path, 'test', 'spam', filename))

        return train_hams, train_spams, test_hams, test_spams

    def word_set(self, filename:list):
        with open(filename, 'r') as f:
            text = f.read()[9:] # Ignoring 'Subject:'
            text = text.replace('\r', '')
            text = text.replace('\n', ' ')
            words = text.split(' ')
            return set(words)

    def fit(self, train_hams:list, train_spams:list):
        """
        :param train_hams: A list of train email filenames which are ham.
        :param train_spams: A list of train email filenames which are spam.
        :return: Nothing.
        """
        def get_counts(filenames:list):
            word_dict = dict()
            for file in filenames:
                words = self.word_set(file)
                for word in words:
                    if word in word_dict:
                        word_dict[word] += 1
                    else:
                        word_dict[word] = 1
            return word_dict
        self.word_counts_ham = get_counts(train_hams)
        self.word_counts_spam = get_counts(train_spams)
        self.num_train_hams = len(train_hams)
        self.num_train_spams = len(train_spams)

    def predict(self, filename:str):
        """
        :param filename: The filename of an email to classify.
        :return: The prediction of Naive Bayes classifier. 
        """

        total = self.num_train_hams + self.num_train_spams
        chance_spam = 0
        chance_ham = 0
        words = self.word_set(filename)
        for word in words:
            chance_spam += np.log((self.word_counts_spam.get(word, 0) + 1) / (self.num_train_spams + 2))
            chance_ham += np.log((self.word_counts_ham.get(word, 0) + 1) / (self.num_train_hams + 2))
        if (self.num_train_hams/total) + chance_ham > (self.num_train_spams/total) + chance_spam:
            return self.HAM_LABEL
        else:
            return self.SPAM_LABEL
            

    def accuracy(self, hams:list, spams:list):
        """
        :param hams: A list of ham email filenames.
        :param spams: A list of spam email filenames.
        :return: The accuracy of our Naive Bayes model.
        """
        total_correct = 0
        total_datapoints = len(hams) + len(spams)
        for filename in hams:
            if self.predict(filename) == self.HAM_LABEL:
                total_correct += 1
        for filename in spams:
            if self.predict(filename) == self.SPAM_LABEL:
                total_correct += 1
        return total_correct / total_datapoints

if __name__ == '__main__':
    # Create a Naive Bayes classifier.
    nbc = NaiveBayes()

    # Load all the train/test ham/spam data.
    train_hams, train_spams, test_hams, test_spams = nbc.load_data()

    # Fit the model to the training data.
    nbc.fit(train_hams, train_spams)

    # Print out the accuracy on the train and test sets.
    print("Train Accuracy: {}".format(nbc.accuracy(train_hams, train_spams)))
    print("Test  Accuracy: {}".format(nbc.accuracy(test_hams, test_spams)))
