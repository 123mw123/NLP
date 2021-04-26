from flask import Flask, render_template, request
import cv2
import numpy as np
import pandas as pd
from load_model import init
import string

app = Flask(__name__)


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


def text_sentence_to_indices(text_sentence, word_to_index):
    tmp = text_sentence.translate(str.maketrans('', '', string.punctuation)).lower().split()
    j = 0
    indices = np.zeros((1, len(tmp)))
    for w in tmp:
        try:
            indices[0, j] = word_to_index[w]
            j = j + 1
        except KeyError:
            pass
    return indices


word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('./model/glove.6B.100d.txt')
model, graph = init()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/prediction', methods=["POST", "GET"])
def prediction():
    print(request.form)
    input_sentence = request.form.get("input")
    with graph.as_default():
        pred = model.predict(text_sentence_to_indices(input_sentence, word_to_index))
        prob = pred[0][0]
        if prob > 0.7:
            output = "Toxic"
        else:
            output = "Not Toxic"
    overall_output = "Output: " + output + " \n" + "Probability: " + str(prob)
    return render_template("prediction.html", data=overall_output)


if __name__ == "__main__":
    app.run(debug=True)
