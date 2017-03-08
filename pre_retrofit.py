from gensim.models import  word2vec


def main():
    model = word2vec.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    with open("vector_file.txt", "w") as vecfile:
        for v in model.vocab:
            vecfile.write("{0} {1}\n".format(v, " ".join(map(str, model[v]))))
    pass

if __name__ == '__main__': main()
