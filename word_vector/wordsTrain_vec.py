# # author:fighting
# # 使用jieba对语料进行词汇分词
# # 使用word2vec对词汇进行向量训练
# import jieba
# import jieba.analyse
# import os
# from gensim.models.word2vec import Word2Vec
# import xlrd
# from langconv import *
# file_path = 'data/LCQMC'
# def getWords(sentences):
#     train_words = []
#     words = []
#     # new_words = open('../data/intention/specialWords', encoding='utf-8').read().split('\n')
#     # for word in new_words:
#     #     jieba.suggest_freq(word, True)
#     for string in sentences:
#         temp = []
#         words_cut = jieba.cut(string, cut_all=False)
#         for word in words_cut:
#             temp.append(word)
#             words.append(word)
#         train_words.append(temp)
#     return words, train_words
#
# # 获取Task1data中的数据
# def get_excel_data(filepath):
#     # 获取数据
#     data = xlrd.open_workbook(filepath)
#     # 获取sheet
#     table = data.sheet_by_name('question')  # 注意读取的表中sheet必须重新命名为question
#     # 获取总行数
#     nrows = table.nrows
#     # 获取总列数
#     nclos = table.ncols
#     ques_list = []
#     for i in range(1, nrows):
#         question = str(table.cell(i, 0).value).strip()
#         if len(question.strip()) >= 1:
#             ques_list.append(question)
#     return ques_list
#
# def Simplified2Traditional(sentences):
#     '''
#     将sentence中的简体字转为繁体字
#     :param sentence: 待转换的句子
#     :return: 将句子中简体字转换为繁体字之后的句子
#     '''
#     sentence_arr = []
#     for sentence in sentences:
#         sentence = Converter('zh-hans').convert(sentence)
#         sentence_arr.append(sentence)
#     print(sentence_arr)
#     return sentence_arr
#
# def get_sentences(file_path):
#     sentences = []
#     for filename in os.listdir(file_path):
#         with open(os.path.join(file_path, filename), 'r') as lcqmc:
#             for line in lcqmc:
#                 linedict = eval(line)
#                 word = linedict['sentence1']
#                 poss = linedict['sentence2']
#                 sentences.append(word)
#                 sentences.append(poss)
#     # texts = open('../gensim_word2vec/data/data_text_noRepeat').read().split('\n')
#     path13 = '../data/public_data/public_rightSen13'
#     path15 = '../data/public_data/public_rightSen15'
#     path_test = '../data/public_data/public_rightTest'
#     path_text = '../data/public_simple_data/data_text'
#     texts13 = open(path13).read().split('\n')
#     texts15 = open(path15).read().split('\n')
#     texts_ = open(path_text).read().split('\n')
#     texts_test = open(path_test).read().split('\n')
#     # print('texts:', texts)
#     public_sens = texts13 + texts15 + texts_test + texts_
#     public_sens = Simplified2Traditional(public_sens)
#     sentences = sentences + public_sens
#     return sentences
#
#
# sentences = get_sentences(file_path)
# for i in range(5):
#     print(sentences[i])
# print("data_text size:", len(sentences))
# words, train_words = getWords(sentences)
#
#
# model = Word2Vec(train_words, size=128, window=4, min_count=1, sg=1, workers=2)
# model.save('wordVec_model/word2vecModel')
# # print(model['鹰潭'])


# author:fighting
from langconv import *
import os
from gensim.models.word2vec import Word2Vec
def Simplified2Traditional(sentence):
    '''
    将sentence中的简体字转为繁体字
    :param sentence: 待转换的句子
    :return: 将句子中简体字转换为繁体字之后的句子
    '''
    sentence = Converter('zh-hans').convert(sentence)
    return sentence


dirname = './data/LCQMC'
sentence = []
words = []
for filename in os.listdir(dirname):
    with open(os.path.join(dirname, filename), 'r') as lcqmc:
        for line in lcqmc:
            linedict = eval(line)
            word = linedict['sentence1']
            poss = linedict['sentence2']
            word = Simplified2Traditional(word)
            pos = Simplified2Traditional(poss)
            sentence.append(word)
            sentence.append(pos)
            sentence.append(poss)

with open('../data/public_simple_data/data_text', 'r') as f:
    for line in f:
        # line = Simplified2Traditional(line)

        sentence.append(line)
print("data_text size:", len(sentence))

path13 = '../data/public_simple_data/rightSen13'
path15 = '../data/public_simple_data/rightSen15'
path_test = '../data/public_simple_data/rightTest'
path_yanzheng_test = '../data/public_simple_data/test_yanzheng_right'
texts13 = open(path13).read().split('\n')
texts15 = open(path15).read().split('\n')
texts_test = open(path_test).read().split('\n')
sentence = sentence + texts13 + texts15 + texts_test
for string in sentence:
    temp = list(string)
    str = ''
    for ch in temp:
        str = str+ch+' '
    # print(str)
    words.append(str)
model = Word2Vec(words, size=128, window=4, min_count=1, sg=1, workers=2)
model.save('wordVec_model/word2vecModel')