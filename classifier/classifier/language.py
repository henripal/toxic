import re
import numpy as np
import torch
from nltk.tokenize import TweetTokenizer


class Language:
    """
    Utility class instanciated using the abstracts; creates
    useful representations of the data for use by models.
    """
    def __init__(self, max_len=50):
        self.vocab_size = 0
        self.vocab = []
        self.labels = ['O', 'B-PROT', 'I-PROT']
        self.label_to_ix = {'O': 0, 'B-PROT': 1, 'I-PROT': 2}
        self.word_to_ix = {}
        self.cleaned_sentences = []
        self.encoded_sentences = None
        self.max_len = max_len

        # used for char:
        self.max_word_len = 15
        self.cleaned_chars = []
        self.encoded_chars = []

        # used for separate test set:
        self.cleaned_test_sentences = []
        self.cleaned_test_chars = []
        self.encoded_test_sentences = None
        self.encoded_test_chars = []

        self.tokenizer = TweetTokenizer(preserve_case = True,
                reduce_len=True,
                strip_handles=True)

    def create_vocab(self):
        vocab = []
        for sentence in self.cleaned_sentences + self.cleaned_test_sentences:
            for word in sentence:
                vocab.append(word)

        self.vocab = list(set(vocab))
        self.vocab.insert(0, 'endofword')
        self.vocab_size = len(self.vocab)

        for i, word in enumerate(self.vocab):
            self.word_to_ix[word] = i

    def clean_word(self, word):
        word = re.sub(r'"', r"", word.lower().strip())
        word = re.sub(r'[0-9]', r'0', word)
        return word

    def clean_char(self, word):
        word = re.sub(r'"', r"", word.strip())
        word = re.sub(r'[0-9]', r'0', word)
        return word

    def clean_sentences(self, sentences):
        """
        populates the class using the corpus as input
        """
        for sentence in sentences:
            sentence = self.tokenize_sentence(sentence)
            out_sentence = []
            labels_sentence = []
            indices_sentence = []
            chars_sentence = []

            for i, word in enumerate(sentence):
                cleaned_word = self.clean_word(word)
                cleaned_char = self.clean_char(word)
                if cleaned_word:
                    out_sentence.append(cleaned_word)
                    chars_sentence.append(cleaned_char)

            self.cleaned_sentences.append(out_sentence)
            self.cleaned_chars.append(chars_sentence)

    def tokenize_sentence(self, sentence):
        """
        tokenizes a sentence
        """
        return self.tokenizer.tokenize(sentence)

    def clean_test_sentences(self, sentences):
        """
        separate method for unlabeled test data.
        """
        for sentence in sentences:
            sentence = self.tokenize_sentence(sentence)
            out_sentence = []
            indices_sentence = []
            chars_sentence = []

            for i, word in enumerate(sentence):
                cleaned_word = self.clean_word(word)
                cleaned_char = self.clean_char(word)
                if cleaned_word:
                    out_sentence.append(cleaned_word)
                    chars_sentence.append(cleaned_char)

            self.cleaned_test_sentences.append(out_sentence)
            self.cleaned_test_chars.append(chars_sentence)

    def encode_sentences(self):
        """
        produces an encoded integer matrix of sentences
        from corpus
        """
        self.encoded_sentences = np.zeros(
            (len(self.cleaned_sentences), self.max_len))

        for i, sentence in enumerate(self.cleaned_sentences):
            sentence_length = min(len(sentence), self.max_len)
            s_array = np.array(list(map(lambda x: self.word_to_ix[x],
                                        sentence)))[:self.max_len]
            self.encoded_sentences[i, :sentence_length] = s_array

        self.encoded_sentences = self.encoded_sentences.astype(int)


        # same for potential test sentences
        self.encoded_test_sentences = np.zeros(
            (len(self.cleaned_test_sentences), self.max_len))

        for i, sentence in enumerate(self.cleaned_test_sentences):
            sentence_length = min(len(sentence), self.max_len)
            s_array = np.array(list(map(lambda x: self.word_to_ix[x],
                                        sentence)))[:self.max_len]
            self.encoded_test_sentences[i, :sentence_length] = s_array

    def encode_chars(self):
        """
        produces an encoded integer matrix of characters from corpus
        encoder is `ord` and decoder is `chr`
        """
        self.encoded_chars = np.zeros(
            (len(self.cleaned_sentences), self.max_len, self.max_word_len))

        for i, sentence in enumerate(self.cleaned_chars):
            sentence = sentence[:self.max_len]
            for j, word in enumerate(sentence):
                word_len = min(len(word), self.max_word_len)
                char_list = []
                for char in word[:word_len]:
                    ordinal = ord(char)
                    if ordinal < 127:
                        char_list.append(ordinal)
                if char_list:
                    self.encoded_chars[i, j, :len(char_list)] = np.array(char_list)
                else:
                    char_list.append(48)
                    self.encoded_chars[i, j, :len(char_list)] = np.array(char_list)

        self.encoded_test_chars = np.zeros(
            (len(self.cleaned_test_sentences), self.max_len, self.max_word_len))

        for i, sentence in enumerate(self.cleaned_test_chars):
            sentence = sentence[:self.max_len]
            for j, word in enumerate(sentence):
                word_len = min(len(word), self.max_word_len)
                char_list = []
                for char in word[:word_len]:
                    ordinal = ord(char)
                    if ordinal < 127:
                        char_list.append(ordinal)
                if char_list:
                    self.encoded_test_chars[i, j, :len(char_list)] = np.array(char_list)
                else:
                    char_list.append(48)
                    self.encoded_test_chars[i, j, :len(char_list)] = np.array(char_list)
        self.encoded_test_chars = self.encoded_test_chars.astype(int)
        self.encoded_chars = self.encoded_chars.astype(int)



    def encode_labels(self):
        """
        encodes labels from IOB to integer
        """
        self.encoded_labels = np.zeros(
            (len(self.cleaned_labels), self.max_len))

        for i, sentence in enumerate(self.cleaned_labels):
            sentence_len = min(len(sentence), self.max_len)
            l_array = np.array(list(map(lambda x: self.label_to_ix[x],
                                        sentence)))[:self.max_len]
            self.encoded_labels[i, :sentence_len] = l_array

    def process_sentences(self, sentences, test_sentences=None):
        """
        main function to process and encode the corpus
        """
        self.clean_sentences(sentences)
        if test_sentences:
            self.clean_test_sentences(test_sentences)
        self.create_vocab()
        self.encode_sentences()
        self.encode_chars()


def create_init_embedding(lang, glove):
    """
    creates the initial embedding matrix from
    glove vectors
    """
    init_embedding = torch.zeros((len(lang.vocab), 50))

    for i, word in enumerate(lang.vocab):
        if word in glove.stoi:
            init_embedding[i, :] = glove.vectors[glove.stoi[word]]

    return init_embedding


