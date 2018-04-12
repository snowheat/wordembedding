from gensim.models import word2vec


def get_sentences_words_array():

    sentences_words_array = []

    with open('id_gsd-ud-train.conllu', 'r', encoding="utf8") as t:

        raw_sentences_array = t.read().split("\n\n")
        for sentences_data in raw_sentences_array:
            sentence = []
            for word_data in sentences_data.splitlines()[2:]:
                sentence.append(word_data.split("\t")[2])

            sentences_words_array.append(sentence)

    return sentences_words_array

def get_word_vector():
    pass

def get_structured_data():
    pass

def train():
    pass

def test():
    pass

print(get_sentences_words_array())
