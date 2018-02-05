import spacy
import numpy as np
class LanguageProcessor:
    def __init__(self):
        pass
    def get_questions_tensor_timeseries(self,questions, nlp, timesteps):
        '''
        Returns a time series of word vectors for tokens in the question
        Input:
        questions: list of unicode objects
        nlp: an instance of the class English() from spacy.en
        timesteps: the number of 
        Output:
        A numpy ndarray of shape: (nb_samples, timesteps, word_vec_dim)
        '''
        nb_samples = len(questions)
        word_vec_dim = nlp(questions[0])[0].vector.shape[0]
        questions_tensor = np.zeros((nb_samples, timesteps, word_vec_dim))
        for i in range(len(questions)):
            tokens = nlp(questions[i])
            for j in range(len(tokens)):
                if j<timesteps:
                    questions_tensor[i,j,:] = tokens[j].vector

        return questions_tensor

    def get_questions_matrix_sum(self,questions,nlp):
        '''
        Sums the word vectors of all the tokens in a question

        Input:
        questions: list of unicode objects
        nlp: an instance of the class English() from spacy.en
        Output:
        A numpy array of shape: (nb_samples, word_vec_dim)	
        '''
        nb_samples = len(questions)
        word_vec_dim = nlp(questions[0])[0].vector.shape[0]
        questions_matrix = np.zeros((nb_samples, word_vec_dim))
        for i in range(len(questions)):
            tokens = nlp(questions[i])
            for j in range(len(tokens)):
                questions_matrix[i,:] += tokens[j].vector

        return questions_matrix
        

    def build_lstm(self):
        pass

    def get_question_features(self,glove_vectors):
        pass