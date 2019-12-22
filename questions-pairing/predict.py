from keras.models import model_from_json
import keras.backend as K
import numpy as np
import string


def text_sentence_to_indices(text_sentence, word_to_index):
    tmp = text_sentence.translate(str.maketrans('','',string.punctuation)).lower().split()
    j = 0
    X_indices = np.zeros((1, len(tmp)))
    for w in tmp:
      try:
        X_indices[0, j] = word_to_index[w]
        j = j+1
      except KeyError:
        pass   
    
    return X_indices


def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map


def is_duplicate(question_1, question_2):
    return np.exp(-np.sum(np.abs(question_1-question_2), axis=1, keepdims=True))


if __name__ == '__main__':

	word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('/home/jason/Documents/glove.6B/glove.6B.300d.txt')

	json_file = open('./sgd_ce_equal.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights("./sgd_ce_equal.h5")
	print("Loaded model from disk")

	input_1 = input("enter first question: ")
	input_2 = input("enter second question: ")

	ind1 = text_sentence_to_indices(input_1,word_to_index)
	ind2 = text_sentence_to_indices(input_2,word_to_index)

	output1 , output2 = loaded_model.predict([ind1,ind2])

	print(is_duplicate(output1,output2))
