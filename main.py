from gensim.models import Word2Vec


def get_sentences_words_array(file_name):

    sentences_words_array = [['']]

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

def get_structured_data(word_vector_model):
    structured_data = []
    sentences, unique_postags = get_sentences_words_postag_array('id_gsd-ud-train.conllu')

    for sentence in sentences[:3]:

        for word_data in sentence:

            if len(word_data) > 1:

                structured_data_item = []

                index = sentence.index(word_data)
                last_index = len(sentence)-1

                if index == 0:
                    for w in word_vector_model['']:
                        structured_data_item.append(w)
                    for w in word_vector_model[sentence[index][0]]:
                        structured_data_item.append(w)
                    for w in word_vector_model[sentence[index+1][0]]:
                        structured_data_item.append(w)
                    for w in get_postags_vector(unique_postags, ''):
                        structured_data_item.append(w)

                if index != 0 and index != last_index:
                    for w in word_vector_model[sentence[index-1][0]]:
                        structured_data_item.append(w)
                    for w in word_vector_model[sentence[index][0]]:
                        structured_data_item.append(w)
                    for w in word_vector_model[sentence[index+1][0]]:
                        structured_data_item.append(w)
                    for w in get_postags_vector(unique_postags, sentence[index-1][1]):
                        structured_data_item.append(w)

                if index == last_index:
                    for w in word_vector_model[sentence[index-1][0]]:
                        structured_data_item.append(w)
                    for w in word_vector_model[sentence[index][0]]:
                        structured_data_item.append(w)
                    for w in word_vector_model['']:
                        structured_data_item.append(w)
                    for w in get_postags_vector(unique_postags, sentence[index-1][1]):
                        structured_data_item.append(w)

                structured_data.append(structured_data_item)

    return structured_data

def train():
    pass

def test():
    pass



word_vector_model = get_word_vector_model(get_sentences_words_array('id_gsd-ud-train.conllu'))
get_structured_data(word_vector_model)