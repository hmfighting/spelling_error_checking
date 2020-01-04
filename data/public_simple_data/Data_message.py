import os
def get_message(path):
    texts = open(path).read().split('\n')
    c_num = 0
    s_num = 0
    sen_num = 0
    for text in texts:
        sen_num += 1
        c = text.count('e')
        c_num = c_num + c
        if 'e' in text:
            s_num = s_num + 1
    return sen_num, c_num, s_num

test_path = './test1_lab'
dev_path = './dev_lab'
train_path = './sentence_lab'

yt_test_path = '../yt_data/data_test'
yt_dev_path = '../yt_data/data_dev'
yt_train_path = '../yt_data/data_tag_test'

yt_test_sen, yt_test_c_num, yt_test_s_num = get_message(yt_test_path)
yt_dev_sen, yt_dev_c_num, yt_dev_s_num = get_message(yt_dev_path)
yt_train_sen, yt_train_c_num, yt_train_s_num = get_message(yt_train_path)


test_sen, test_c_num, test_s_num = get_message(test_path)
dev_sen, dev_c_num, dev_s_num = get_message(dev_path)
train_sen, train_c_num, train_s_num = get_message(train_path)


print('test_sen:', test_sen, 'test_c_num:', str(test_c_num), 's_num:', str(test_s_num))
print('dev_sen:', dev_sen, 'dev_c_num:', str(dev_c_num), 'dev_s_num:', str(dev_s_num))
print('train_sen', train_sen, 'train_c_num:', str(train_c_num), 'train_s_num:', str(train_s_num))

print('yt_test_sen:', yt_test_sen, 'yt_test_c_num:', str(yt_test_c_num), 'yt_s_num:', str(yt_test_s_num))
print('yt_dev_sen:', yt_dev_sen, 'yt_dev_c_num:', str(yt_dev_c_num), 'yt_dev_s_num:', str(yt_dev_s_num))
print('yt_train_sen', yt_train_sen, 'yt_train_c_num:', str(yt_train_c_num), 'yt_train_s_num:', str(yt_train_s_num))


dirname = '/Users/fighting/PycharmProjects/Text_error_detection/word_vector/data/LCQMC'
sentence = []
words = []
cqmc_num = 0
for filename in os.listdir(dirname):
    with open(os.path.join(dirname, filename), 'r') as lcqmc:
        for line in lcqmc:
            cqmc_num += 1
print('CQMC_size:', cqmc_num)


