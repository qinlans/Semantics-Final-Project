from gensim.models import  word2vec
from collections import  defaultdict

def main():
    model = word2vec.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    with open("vector_file.txt", "w") as vecfile:
        for v in model.vocab:
            vecfile.write("{0} {1}\n".format(v, " ".join(map(str, model[v]))))
    pass
def main2():
    model = word2vec.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    model.save_word2vec_format(fname='vector_file.txt', binary=False)
    pass

def read_file(filename):
  dataset = []
  with open(filename, 'r') as f:
    for line in f:
      sent = line.strip().split(' ')
      sent.append('</S>')
      sent = ['<S>'] + sent
      dataset.append(sent)
  return dataset



def build_dicts(corpus, unk_threshold=1):
    word_counts = defaultdict(lambda: 0)
    for line in corpus:
        for word in line:
            word_counts[word] += 1

    token_to_id = defaultdict(lambda: 0)
    token_to_id['UNK'] = 0
    token_to_id['<S>'] = 1
    token_to_id['</S>'] = 2

    id_to_token = ['UNK', '<S>', '</S>']

    for word, count in word_counts.items():
        if count > unk_threshold and not word in token_to_id:
            token_to_id[word] = len(token_to_id)
            id_to_token.append(word)

    return token_to_id, id_to_token

def main_filtered():

    training_src = read_file("/Users/elliotschumacher/Dropbox/git/Semantics-Final-Project/en-fr/train.low.filt.en")
    token_to_id, _ = build_dicts(training_src)

    word_list = list(token_to_id.keys())

    vector_dict = {}

    model = word2vec.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    for v in word_list:
        if v in model.vocab:
            vector_dict[v] = model[v]

    with open("vector_file_enfr.txt", "w") as vecfile:

        model_vocab_size = len(vector_dict)

        vecfile.write("{0} {1}\n".format(model_vocab_size,  model.vector_size))
        count = 0
        for v in vector_dict:
            vecfile.write("{0} {1}\n".format(v, " ".join(map(str, model[v]))))
            if count % 1000 == 0:
                print("{0} / {1}".format(count, model_vocab_size))
            count += 1

def main_all():

    training_src = read_file("/Users/elliotschumacher/Dropbox/git/Semantics-Final-Project/en-fr/train.low.filt.en")
    token_to_id, _ = build_dicts(training_src)

    word_list = list(token_to_id.keys())

    vector_dict = {}

    model = word2vec.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    for v in word_list:
        if v in model.vocab:
            vector_dict[v] = model[v]

    with open("vector_file_enfr.txt", "w") as vecfile:

        model_vocab_size = len(vector_dict)

        vecfile.write("{0} {1}\n".format(model_vocab_size,  model.vector_size))
        count = 0
        for v in vector_dict:
            vecfile.write("{0} {1}\n".format(v, " ".join(map(str, model[v]))))
            if count % 1000 == 0:
                print("{0} / {1}".format(count, model_vocab_size))
            count += 1

if __name__ == '__main__': main3()
