from gensim.models import Word2Vec
from neural_net import NeuralNet

predictor = NeuralNet()


def get_sentences_words_array(file_names):
    sentences_words_array = [['']]

    for file_name in file_names:
        with open(file_name, 'r', encoding="utf8") as t:

            raw_sentences_array = t.read().split("\n\n")
            for sentences_data in raw_sentences_array:
                sentence = []
                for word_data in sentences_data.splitlines()[2:]:
                    sentence.append(word_data.split("\t")[2])

                sentences_words_array.append(sentence)

    return sentences_words_array


def get_sentences_words_postag_array(file_name):
    sentences_words_postag_array = [['']]
    unique_postags = []

    with open(file_name, 'r', encoding="utf8") as t:

        raw_sentences_array = t.read().split("\n\n")
        for sentences_data in raw_sentences_array:
            sentence = []
            for word_data in sentences_data.splitlines()[2:]:
                word_data_items = word_data.split("\t")
                sentence.append([word_data_items[2], word_data_items[3]])
                if word_data_items[3] not in unique_postags:
                    unique_postags.append(word_data_items[3])

            sentences_words_postag_array.append(sentence)

    return sentences_words_postag_array, unique_postags


def get_word_vector_model(sentences):
    model = Word2Vec(sentences, size=100, min_count=1, window=5, workers=4, sg=0)
    return model


def get_postags_vector(postags, input_postag):
    postags_vector = []
    for postag in postags:
        if postag == input_postag:
            postags_vector.append(1)
        else:
            postags_vector.append(0)
    return postags_vector


def get_structured_data(word_vector_model, filename):
    structured_data = []
    label = []
    sentences, unique_postags = get_sentences_words_postag_array(filename)

    for sentence in sentences[:]:

        for word_data in sentence:

            if len(word_data) > 1:

                structured_data_item = []

                index = sentence.index(word_data)
                last_index = len(sentence) - 1

                if index == 0:
                    for w in word_vector_model['']:
                        structured_data_item.append(w)
                    for w in word_vector_model[sentence[index][0]]:
                        structured_data_item.append(w)
                    for w in word_vector_model[sentence[index + 1][0]]:
                        structured_data_item.append(w)
                    for w in get_postags_vector(unique_postags, ''):
                        structured_data_item.append(w)

                if index != 0 and index != last_index:
                    for w in word_vector_model[sentence[index - 1][0]]:
                        structured_data_item.append(w)
                    for w in word_vector_model[sentence[index][0]]:
                        structured_data_item.append(w)
                    for w in word_vector_model[sentence[index + 1][0]]:
                        structured_data_item.append(w)
                    for w in get_postags_vector(unique_postags, sentence[index - 1][1]):
                        structured_data_item.append(w)

                if index == last_index:
                    for w in word_vector_model[sentence[index - 1][0]]:
                        structured_data_item.append(w)
                    for w in word_vector_model[sentence[index][0]]:
                        structured_data_item.append(w)
                    for w in word_vector_model['']:
                        structured_data_item.append(w)
                    for w in get_postags_vector(unique_postags, sentence[index - 1][1]):
                        structured_data_item.append(w)

                structured_data.append(structured_data_item)
                label.append(sentence[index][1])

    return structured_data, label


def train(model):
    data, label = get_structured_data(model, 'id_gsd-ud-train.conllu')
    predictor.train(data, label)

    pass


def test(model):
    data, label = get_structured_data(model, 'id_gsd-ud-test.conllu')
    predictor.test(data, label)
    pass


sentences = get_sentences_words_array(['id_gsd-ud-train.conllu', 'id_gsd-ud-test.conllu'])
print("sentences")

word_vector_model = get_word_vector_model(sentences)
print("word2vec")

train(word_vector_model)
print("train")

test(word_vector_model)
print("test")
