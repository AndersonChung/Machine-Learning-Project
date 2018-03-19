import jieba
import pandas as pd
import numpy as np


def cut_sentence(data, save_path):
    total = len(data)
    i = 1
    preprocess_data = list()
    for sentence in data.as_matrix():
        print("{0:.2f}".format(i / total))
        i += 1
        # s = ''.join(filter(lambda x: x not in punct, sentence[0]))
        seg_list = jieba.cut(sentence[0])
        seg_list = ("|".join(seg_list)).split('|')
        preprocess_data.append(seg_list)
    np.save(save_path, preprocess_data)


def cut_sentence_clip_to_350(data, save_path):
    total = len(data)
    i = 1
    preprocess_data = list()
    for sentence in data.as_matrix():
        print("{0:.2f}".format(i / total))
        i += 1
        # s = ''.join(filter(lambda x: x not in punct, sentence[0]))
        seg_list = jieba.cut(sentence[0])
        seg_list = ("|".join(seg_list)).split('|')
        preprocess_data.append(seg_list[0:350])
    np.save(save_path, preprocess_data)


def train_span_dummy(train_span, max_index):
    def dummy_vector(vec, max_index):
        dum_v = np.zeros(max_index + 1)
        for i in range(vec[0], vec[1] + 1):
            dum_v[i] = 1
        return dum_v
    
    dum_span = np.array([dummy_vector(t, max_index) for t in train_span])
    print('shape of dummy_span : ', dum_span.shape)
    np.save('./Preprocess_data/dummy_span.npy', dum_span)


def train_span_start_length(train_span, max_index):
    def dummy_vector(vec, max_index):
        dum_v = np.zeros(max_index + 1)
        dum_v[vec[0]] = 1
        return dum_v
    
    def length(vec):
        return vec[1] - vec[0] + 1
    
    def dummy_end(vec, max_index):
        dum_v = np.zeros(max_index + 1)
        dum_v[vec[1]] = 1
        return dum_v
    
    dum_span_end = np.array([dummy_end(t, max_index) for t in train_span])
    dum_span_start = np.array([dummy_vector(t, max_index) for t in train_span])
    dum_span_length = np.array([length(t) for t in train_span])
    print('shape of dum_span_start:', dum_span_start.shape,
          '\nshape of dum_span_end: ', dum_span_end.shape,
          '\nshape of dum_span_length: ', dum_span_length[1:10])
    
    np.save('./Preprocess_data/dum_wordspan_start.npy', dum_span_start)
    np.save('./Preprocess_data/dum_wordspan_end.npy', dum_span_end)
    np.save('./Preprocess_data/dum_wordspan_length.npy', dum_span_length)


def word_to_character(word_Length, wordID, questionID):
    wordID_start = wordID[0]
    wordID_end = wordID[1]
    characterID_start = sum(word_Length[questionID][0:(wordID_start)])
    characterID_end = characterID_start + \
                      sum(word_Length[questionID][(wordID_start):(wordID_end + 1)]) - 1
    return ([characterID_start, characterID_end])


def character_to_word(characterID, questionID, word_length):
    characterID_start = characterID[0]
    characterID_end = characterID[1]
    wordID_start = 0
    summation_start = 0
    wordID_end = 0
    summation_end = 0
    while (summation_start <= characterID_start):
        summation_start += word_length[questionID][wordID_start]
        wordID_start += 1
    while (summation_end <= characterID_end):
        summation_end += word_length[questionID][wordID_end]
        wordID_end += 1
    return ([wordID_start - 1, wordID_end - 1])


def train_word_span(train_span, train_context):
    word_length = []
    for item in train_context:
        word_length.append([len(i) for i in item])
    word_span = []
    for i in range(train_span.shape[0]):
        word_span.append(character_to_word(train_span[i], i, word_length))
    return np.array(word_span)


def train_word_span_decode(train_prediction_span, train_context):
    word_length = []
    for item in train_context:
        word_length.append([len(i) for i in item])
    word_span = []
    for i in range(train_span.shape[0]):
        word_span.append(word_to_character(train_prediction_span[i], i, word_length))
    return np.array(word_span)


def set_word_length_list(context):
    word_length_list = []
    for paragraph in context:
        word_length_list.append([len(i) for i in paragraph])
    return (word_length_list)


# word_Length = set_word_length_list(train_context)
# test_word_Length = set_word_length_list(test_context)


if __name__ == '__main__':
    test_question = pd.read_csv('./data/test.question', sep='delimiter', header=None, encoding='utf-8')
    test_context = pd.read_csv('./data/testl.context', sep='delimiter', header=None, encoding='utf-8')
    train_context = pd.read_csv('./data/train.context', sep='delimiter', header=None, encoding='utf-8')
    train_span = pd.read_csv('./data/train.span', sep='delimiter', header=None, encoding='utf-8')
    train_question = pd.read_csv('./data/train.question', sep='delimiter', header=None, encoding='utf-8')
    train_span_save = np.array([x[0].split(' ') for x in train_span.as_matrix()]).astype('int')
    np.save('./Preprocess_data/train_span.npy', train_span_save)
    train_span = np.load('./Preprocess_data/train_span.npy')
    # cut sentences to lists of words and save
    cut_sentence(train_context, './Preprocess_data/train_context.npy')
    cut_sentence(train_question, './Preprocess_data/train_question.npy')
    cut_sentence(test_context, './Preprocess_data/test_context.npy')
    cut_sentence(test_question, './Preprocess_data/test_qusetion.npy')
    cut_sentence_clip_to_350(train_context, './Preprocess_data/train_context_350.npy')
    cut_sentence_clip_to_350(test_context, './Preprocess_data/test_context_350.npy')
    # read lists of words
    cut_train_context = np.load('./Preprocess_data/train_context.npy')
    cut_train_question = np.load('./Preprocess_data/train_question.npy')
    cut_test_context = np.load('./Preprocess_data/test_context.npy')
    cut_test_question = np.load('./Preprocess_data/test_qusetion.npy')
    cut_train_context_350 = np.load('./Preprocess_data/train_context_350.npy')
    cut_test_context_350 = np.load('./Preprocess_data/test_context_350.npy')

    # convert the training character span to chunks of word span and clip to (0,349)

    word_span = train_word_span(train_span, cut_train_context)
    word_span = np.clip(word_span, 0, 349)
    length_for_each_train_paragraph = [len(P) for P in cut_train_context_350]
    length_for_each_train_paragraph.sort()
    length_for_each_train_paragraph[-1]
    np.save('./Preprocess_data/word_span.npy', word_span)
    # one-hot encoding for chunks of word span
    train_span_dummy(word_span, 349)
    train_span_start_length(word_span, 349)


    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    def loss_list(data):
        loss_list = []
        for cut in range(np.max(data)):
            loss_list.append(len([i for i in data if i > cut]) / len(data))
        return(loss_list)

    test_question = pd.read_csv('./data/test.question', sep='delimiter', header=None, encoding='utf-8')
    test_context = pd.read_csv('./data/testl.context', sep='delimiter', header=None, encoding='utf-8')
    train_context = pd.read_csv('./data/train.context', sep='delimiter', header=None, encoding='utf-8')
    train_span = np.load('./Preprocess_data/train_span.npy')
    word_span = train_word_span(train_span, cut_train_context)
    train_question = pd.read_csv('./data/train.question', sep='delimiter', header=None, encoding='utf-8')
    cut_train_context  = np.load('./Preprocess_data/train_context.npy' )
    cut_train_question = np.load('./Preprocess_data/train_question.npy')
    cut_test_context   = np.load('./Preprocess_data/test_context.npy'  )
    cut_test_question  = np.load('./Preprocess_data/test_qusetion.npy' )
    T = [len(i) for i in cut_train_context]
    E = [len(i) for i in cut_test_context]
    W = [i[0] for i in train_span]
    D = [i[0] for i in word_span]
    train_span = np.load('./Preprocess_data/train_span.npy')
    # plot cut_train_context
    T_plot = plt.hist(T, normed=True,bins = 300)
    plt.show()
    # plot cut_test_context
    E_plot = plt.hist(E, normed=True, bins = 300)
    plt.show()
    # plot train_span
    W_plot = plt.hist(W, normed=True, bins = 300)
    plt.show()
    np.save('T', T)



def loss_list(data):
    loss_list = []
    for cut in range(np.max(data)):
        loss_list.append(len([i for i in data if i > cut]) / len(data))
    return(loss_list)


T_loss_list = np.subtract(1,loss_list(T))
E_loss_list = np.subtract(1,loss_list(E))
W_loss_list = loss_list(W)
D_loss_list = loss_list(D)

T_plot = plt.scatter(range(1, 688), T_loss_list)
plt.xlabel("number of train contexts' word count ")
plt.ylabel('cumulative frequency')
plt.title('Train context length cumulative plot')
plt.show()

E_plot = plt.scatter(range(1, len(E_loss_list) + 1), E_loss_list)
plt.xlabel("number of test contexts' word count ")
plt.ylabel('cumulative frequency')
plt.title('Test context length cumulative plot')
plt.show()

W_plot = plt.scatter(range(1, len(W_loss_list) + 1), W_loss_list, alpha=0.5)
plt.xlabel("True answer's character starting position")
plt.ylabel('cumulative frequency')
plt.title('Train span of location of character')
plt.show()



D_plot = plt.scatter(range(1, len(D_loss_list) + 1), D_loss_list)
plt.show()
