from collections import defaultdict
import dynet as dy
import numpy as np
import random
import re
import sys
import pickle
import os
trans_out_dir = './output/'

LOAD_MODEL = False 
TRAIN = True

def read_file(filename):
  dataset = []
  with open(filename, 'r') as f:
    for line in f:
      sent = line.strip().split(' ')
      sent.append('</S>')
      sent = ['<S>'] + sent
      dataset.append(sent)
  return dataset

def build_dicts(corpus, unk_threshold=1, vector_word_list=None):
    word_counts = defaultdict(lambda: 0)
    for line in corpus:
        for word in line:
            word_counts[word] += 1

    if vector_word_list is None:
        token_to_id = defaultdict(lambda: 0)
        token_to_id['UNK'] = 0
        token_to_id['<S>'] = 1
        token_to_id['</S>'] = 2
        token_sense_to_id = None

        id_to_token = ['UNK', '<S>', '</S>']
        for word, count in word_counts.items():
            if count > unk_threshold and not word in token_to_id:
                token_to_id[word] = len(token_to_id)
                id_to_token.append(word)
    else:
        token_to_id = defaultdict(lambda: [0])
        token_to_id['UNK'] = [0]
        token_to_id['<S>'] = [1]
        token_to_id['</S>'] = [2]

        #Note : I need this dict to load the sense vectors into the right place.  Keeping original as well
        token_sense_to_id = defaultdict(lambda: 0)
        token_sense_to_id[('UNK',0)] = 0
        token_sense_to_id[('<S>',0)] = 1
        token_sense_to_id[('</S>',0)] = 2

        #Included sense number for reference to the sense produced by Sense Retrofit
        id_to_token = {0: ('UNK', 0), 1: ('<S>', 0), 2: ('</S>', 0)}

        num_tokens = len(id_to_token)

        for word, count in word_counts.items():
            if count > unk_threshold and word not in token_to_id and word in vector_word_list:
                token_to_id[word] = []

                for sense in vector_word_list[word]:
                    token_to_id[word].append(num_tokens)
                    token_sense_to_id[(word, sense)] = num_tokens

                    id_to_token[num_tokens] = (word, sense)
                    num_tokens += 1
            elif count > unk_threshold and word not in token_to_id and word not in vector_word_list:
                token_to_id[word] = []

                token_to_id[word].append(num_tokens)
                token_sense_to_id[(word, 0)] = num_tokens

                id_to_token[num_tokens] = (word, 0)
                num_tokens += 1


    return token_to_id, id_to_token, token_sense_to_id

# Creates batches where all source sentences are the same length
def create_batches(sorted_dataset, max_batch_size):
    source = [x[0] for x in sorted_dataset]
    src_lengths = [len(x) for x in source]
    batches = []
    prev = src_lengths[0]
    prev_start = 0
    batch_size = 1
    for i in range(1, len(src_lengths)):
        if src_lengths[i] != prev or batch_size == max_batch_size:
            batches.append((prev_start, batch_size))
            prev = src_lengths[i]
            prev_start = i
            batch_size = 1
        else:
            batch_size += 1
    return batches

class Attention:
    def __init__(self, model, training_src, training_tgt, model_name, max_batch_size=32, num_epochs=30,
            layers=1, embed_size=300, hidden_size=512, word_attention_size=128, sense_attention_size=32,
            max_len=50, unk_threshold=1, src_vectors_file=None, builder=dy.LSTMBuilder):
        self.model = model
        self.training = [(x, y) for (x, y) in zip(training_src, training_tgt)]

        vector_word_list = self.load_src_words(src_vectors_file)
        print ('Building dictionaries for converting from ids to tokens and vice versa')
        self.src_token_to_id, self.src_id_to_token, self.src_token_sense_to_id = build_dicts(training_src, unk_threshold, vector_word_list)

        self.src_vocab_size = len(self.src_id_to_token)
        self.tgt_token_to_id, self.tgt_id_to_token, _ = build_dicts(training_tgt, unk_threshold)
        self.tgt_vocab_size = len(self.tgt_token_to_id)
        self.model_name = model_name
        self.max_batch_size = max_batch_size
        self.num_epochs = num_epochs
        self.layers = layers
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.word_attention_size = word_attention_size
        self.sense_attention_size = sense_attention_size
        self.max_len = max_len

        if src_vectors_file is not None:
            self.load_src_lookup_params(src_vectors_file, model)
        else:
            self.src_lookup = model.add_lookup_parameters((self.src_vocab_size, self.embed_size))

        self.tgt_lookup = model.add_lookup_parameters((self.tgt_vocab_size, self.embed_size))
        self.l2r_builder = builder(self.layers, self.embed_size, self.hidden_size, model)
        self.r2l_builder = builder(self.layers, self.embed_size, self.hidden_size, model)

        self.sense_builder = builder(self.layers, self.embed_size*2, self.hidden_size, model)
        self.word_dec_builder = builder(self.layers, self.hidden_size*2+self.embed_size, self.hidden_size, model)

        self.W_s = model.add_parameters((self.hidden_size, self.hidden_size*2))
        self.b_s = model.add_parameters((self.hidden_size))

        self.W_m = model.add_parameters((self.hidden_size, self.hidden_size*3))
        self.b_m = model.add_parameters((self.hidden_size))

        self.W_y = model.add_parameters((self.tgt_vocab_size, self.hidden_size))
        self.b_y = model.add_parameters((self.tgt_vocab_size))

        self.W1_att_f = model.add_parameters((self.word_attention_size, self.hidden_size * 2))
        self.W1_att_e = model.add_parameters((self.word_attention_size, self.hidden_size))
        self.w2_att = model.add_parameters((self.word_attention_size))

        self.W1_att_senses = model.add_parameters((self.sense_attention_size, self.embed_size))
        self.W1_att_m = model.add_parameters((self.sense_attention_size, self.hidden_size))
        self.w2_att_s = model.add_parameters((self.sense_attention_size))

    def load_src_words(self, src_vectors_file):
        print('Reading source vectors from file ' + src_vectors_file)
        if src_vectors_file is None:
            return None
        else:
            word_list = defaultdict(lambda: list())
            count = 0
            with open(src_vectors_file) as vector_file:
                first_line = True
                for l in vector_file:
                    if first_line:
                        first_line = False
                    else:
                        try:
                            space_delim = l.split()
                            word = space_delim[0].split('|')[0]
                            sense = space_delim[0].split('|')[1].strip(':')
                            word_list[word].append(sense)
                        except Exception as e:
                            print('Error:{0}, {1}'.format(e, l))
                    count += 1

            return word_list

    def load_src_lookup_params_only_vectors(self, src_vectors_file, model):
        print('Loading source vectors as lookup parameters')
        init_array = np.zeros((self.src_vocab_size, self.embed_size))
        count = 0
        with open(src_vectors_file) as vector_file:
            first_line = True
            for l in vector_file:
                if first_line:
                    first_line = False
                else:
                    try:
                        space_delim = l.split()
                        word = space_delim[0].split('|')[0]
                        sense = space_delim[0].split('|')[1].strip(':')
                        w_id = int(self.src_token_sense_to_id[(word, sense)])
                        if w_id != 0:
                            init_array[w_id, :] = np.asarray(space_delim[1:])
                            count += 1

                    except Exception as e:
                        print('Error:{0}, {1}'.format(e, l))


        print('Set: {0} vectors out of vocab size: {1}'.format(count, self.src_vocab_size))
        self.src_lookup = model.add_lookup_parameters((self.src_vocab_size, self.embed_size))

        self.src_lookup.init_from_array(init_array)

    def load_src_lookup_params(self, src_vectors_file, model):
        self.src_lookup = model.add_lookup_parameters((self.src_vocab_size, self.embed_size))

        pickle_fn = 'src_lookup_vectors_retro.pkl'
        print('Loading source vectors as lookup parameters')
        count = 0

        if not os.path.exists(pickle_fn):
            init_array = np.zeros((self.src_vocab_size, self.embed_size))

            with open(src_vectors_file) as vector_file:
                first_line = True
                for l in vector_file:
                    if first_line:
                        first_line = False
                    else:
                        try:
                            space_delim = l.split()
                            word = space_delim[0].split('|')[0]
                            sense = space_delim[0].split('|')[1].strip(':')
                            w_id = int(self.src_token_sense_to_id[(word, sense)])
                            if w_id != 0:
                                init_array[w_id, :] = np.asarray(space_delim[1:])
                                count += 1

                        except Exception as e:
                            print('Error:{0}, {1}'.format(e, l))
            with open(pickle_fn, 'wb') as pickle_file:
                pickle.dump(init_array, pickle_file)
            for i in range(self.src_vocab_size):
                if not np.any(init_array[i, :]):
                    expr = dy.lookup(self.src_lookup, i)
                    init_array[i, :] = expr.npvalue()
        else:
            with open(pickle_fn, 'rb') as pickle_file:
                init_array = pickle.load(pickle_file)

            for i in range(self.src_vocab_size):
                if not np.any(init_array[i, :]):
                    expr = dy.lookup(self.src_lookup, i)
                    init_array[i, :] = expr.npvalue()
                else:
                    count += 1

        print('Set: {0} vectors out of vocab size: {1}'.format(count, self.src_vocab_size))

        self.src_lookup.init_from_array(init_array)

    def save_model(self):
        self.model.save(self.model_name)

    def load_model(self):
        self.model.load(self.model_name)

    # Calculates the generalized vector over senses using a MLP
    def __sense_attention_mlp(self, h_senses, h_m):
        W1_att_senses = dy.parameter(self.W1_att_senses)
        W1_att_m = dy.parameter(self.W1_att_m)
        w2_att_s = dy.parameter(self.w2_att_s)
        a_t = dy.transpose(dy.tanh(dy.colwise_add(W1_att_senses * h_senses, W1_att_m * h_m))) * w2_att_s
        alignment = dy.softmax(a_t)
        c_t = h_senses * alignment
        return c_t

    # Calculates the context vector for words using a MLP
    def __word_attention_mlp(self, h_fs_matrix, h_e, fixed_attentional_component):
        W1_att_e = dy.parameter(self.W1_att_e)
        w2_att = dy.parameter(self.w2_att)
        a_t = dy.transpose(dy.tanh(dy.colwise_add(fixed_attentional_component, W1_att_e * h_e))) * w2_att
        alignment = dy.softmax(a_t)
        c_t = h_fs_matrix * alignment
        return c_t

    # Training step over a single sentence pair
    def __step(self, instance):
        dy.renew_cg()

        W_s = dy.parameter(self.W_s)
        b_s = dy.parameter(self.b_s)
        W_y = dy.parameter(self.W_y)
        b_y = dy.parameter(self.b_y)
        W_m = dy.parameter(self.W_m)
        b_m = dy.parameter(self.b_m)
        W1_att_f = dy.parameter(self.W1_att_f)
        W1_att_e = dy.parameter(self.W1_att_e)
        w2_att = dy.parameter(self.w2_att)

        src_sent, tgt_sent = instance

        # Sense-level attention
        attended = []
        c_t_sense = dy.vecInput(self.embed_size)
        sense_start = dy.concatenate([dy.lookup(self.src_lookup, self.src_token_to_id['<S>'][0]), dy.tanh(c_t_sense)])
        sense_state = self.sense_builder.initial_state().add_input(sense_start)
        for cw in src_sent:
            cw_sense_ids = self.src_token_to_id[cw]
            cw_senses = [dy.lookup(self.src_lookup, sense_id) for sense_id in cw_sense_ids]
            h_senses = dy.concatenate_cols(cw_senses)
            h_m = sense_state.output()
            c_t_sense = self.__sense_attention_mlp(h_senses, h_m)
            sense_state = sense_state.add_input(dy.concatenate([c_t_sense, dy.tanh(c_t_sense)]))
            attended.append(c_t_sense)

        attended_rev = list(reversed(attended))

        # Bidirectional representations
        l2r_state = self.l2r_builder.initial_state()
        r2l_state = self.r2l_builder.initial_state()
        l2r_contexts = []
        r2l_contexts = []
        for (cw_l2r, cw_r2l) in zip(attended, attended_rev):
            l2r_state = l2r_state.add_input(cw_l2r)
            r2l_state = r2l_state.add_input(cw_r2l)
            l2r_contexts.append(l2r_state.output())
            r2l_contexts.append(r2l_state.output())
        r2l_contexts.reverse()

        h_fs = []
        for (l2r_i, r2l_i) in zip(l2r_contexts, r2l_contexts):
            h_fs.append(dy.concatenate([l2r_i, r2l_i]))
        h_fs_matrix = dy.concatenate_cols(h_fs)
        fixed_attentional_component = W1_att_f * h_fs_matrix 

        losses = []
        num_words = 0

        # Decoder
        c_t = dy.vecInput(self.hidden_size * 2)
        start_state = dy.affine_transform([b_s, W_s, h_fs[-1]])
        dec_state = self.word_dec_builder.initial_state().set_s([start_state, dy.tanh(start_state)])
        for (cw, nw) in zip(tgt_sent, tgt_sent[1:]):
            embed_t = dy.lookup(self.tgt_lookup, self.tgt_token_to_id[cw])
            x_t = dy.concatenate([embed_t, c_t])
            dec_state = dec_state.add_input(x_t)
            h_e = dec_state.output()
            c_t = self.__word_attention_mlp(h_fs_matrix, h_e, fixed_attentional_component)
            m_t = dy.tanh(dy.affine_transform([b_m, W_m, dy.concatenate([h_e, c_t])]))
            y_star = dy.affine_transform([b_y, W_y, m_t])
            loss = dy.pickneglogsoftmax(y_star, self.tgt_token_to_id[nw])
            losses.append(loss)
            num_words += 1
 
        return dy.esum(losses), num_words

    def translate_sentence(self, sent):
        dy.renew_cg()

        W_s = dy.parameter(self.W_s)
        b_s = dy.parameter(self.b_s)
        W_y = dy.parameter(self.W_y)
        b_y = dy.parameter(self.b_y)
        W_m = dy.parameter(self.W_m)
        b_m = dy.parameter(self.b_m)
        W1_att_f = dy.parameter(self.W1_att_f)
        w2_att = dy.parameter(self.w2_att)

        # Sense-level attention
        attended = []
        c_t_sense = dy.vecInput(self.embed_size)
        sense_start = dy.concatenate([dy.lookup(self.src_lookup, self.src_token_to_id['<S>'][0]), c_t_sense])
        sense_state = self.sense_builder.initial_state().add_input(sense_start)
        for cw in sent:
            cw_sense_ids = self.src_token_to_id[cw]
            cw_senses = [dy.lookup(self.src_lookup, sense_id) for sense_id in cw_sense_ids]
            h_senses = dy.concatenate_cols(cw_senses)
            h_m = sense_state.output()
            c_t_sense = self.__sense_attention_mlp(h_senses, h_m)
            sense_state = sense_state.add_input(dy.concatenate([c_t_sense, dy.tanh(c_t_sense)]))
            attended.append(c_t_sense)

        attended_rev = list(reversed(attended))

        # Bidirectional representations
        l2r_state = self.l2r_builder.initial_state()
        r2l_state = self.r2l_builder.initial_state()
        l2r_contexts = []
        r2l_contexts = []
        for (cw_l2r, cw_r2l) in zip(attended, attended_rev):
            l2r_state = l2r_state.add_input(cw_l2r)
            r2l_state = r2l_state.add_input(cw_r2l)
            l2r_contexts.append(l2r_state.output())
            r2l_contexts.append(r2l_state.output())
        r2l_contexts.reverse()

        h_fs = []
        for (l2r_i, r2l_i) in zip(l2r_contexts, r2l_contexts):
            h_fs.append(dy.concatenate([l2r_i, r2l_i]))
        h_fs_matrix = dy.concatenate_cols(h_fs)
        fixed_attentional_component = W1_att_f * h_fs_matrix 

        # Decoder
        trans_sentence = ['<S>']
        cw = trans_sentence[-1]
        c_t = dy.vecInput(self.hidden_size * 2)
        start_state = dy.affine_transform([b_s, W_s, h_fs[-1]])
        dec_state = self.word_dec_builder.initial_state().set_s([start_state, dy.tanh(start_state)])
        while len(trans_sentence) < self.max_len:
            embed_t = dy.lookup(self.tgt_lookup, self.tgt_token_to_id[cw])
            x_t = dy.concatenate([embed_t, c_t])
            dec_state = dec_state.add_input(x_t)
            h_e = dec_state.output()
            c_t = self.__word_attention_mlp(h_fs_matrix, h_e, fixed_attentional_component)
            m_t = dy.tanh(dy.affine_transform([b_m, W_m, dy.concatenate([h_e, c_t])]))
            y_star = dy.affine_transform([b_y, W_y, m_t])
            p = dy.softmax(y_star)
            cw = self.tgt_id_to_token[np.argmax(p.vec_value())]
            if cw == '</S>':
                break
            trans_sentence.append(cw)

        return ' '.join(trans_sentence[1:])

    def __step_batch(self, batch):
        dy.renew_cg()

        W_s = dy.parameter(self.W_s)
        b_s = dy.parameter(self.b_s)
        W_y = dy.parameter(self.W_y)
        b_y = dy.parameter(self.b_y)
        W_m = dy.parameter(self.W_m)
        b_m = dy.parameter(self.b_m)
        W1_att_f = dy.parameter(self.W1_att_f)
        w2_att = dy.parameter(self.w2_att)

        src_batch = [x[0] for x in batch]
        tgt_batch = [x[1] for x in batch]
        batch_size = len(src_batch)

        attended_batch = []
        for src_sent in src_batch:
            attended = []
            c_t_sense = dy.vecInput(self.embed_size)
            sense_start = dy.concatenate([dy.lookup(self.src_lookup, self.src_token_to_id['<S>'][0]), dy.tanh(c_t_sense)])
            sense_state = self.sense_builder.initial_state().add_input(sense_start)

            for cw in src_sent:
                cw_sense_ids = self.src_token_to_id[cw]
                cw_senses = [dy.lookup(self.src_lookup, sense_id) for sense_id in cw_sense_ids]
                h_senses = dy.concatenate_cols(cw_senses)
                h_m = sense_state.output()
                c_t_sense = self.__sense_attention_mlp(h_senses, h_m)
                sense_state = sense_state.add_input(dy.concatenate([c_t_sense, dy.tanh(c_t_sense)]))
                attended.append(c_t_sense)

            attended_batch.append(attended)
        attended_batch_rev = [list(reversed(sent)) for sent in attended_batch]

        # Encoder
        src_cws_l2r = []
        src_cws_r2l = []
        src_len = [len(sent) for sent in attended_batch]        
        max_src_len = np.max(src_len)
 
        for i in range(max_src_len):
            src_cws_l2r.append([sent[i] for sent in attended_batch])
            src_cws_r2l.append([sent[i] for sent in attended_batch_rev])

        l2r_state = self.l2r_builder.initial_state()
        r2l_state = self.r2l_builder.initial_state()
        l2r_contexts = []
        r2l_contexts = []
        for i, (cws_l2r, cws_r2l) in enumerate(zip(src_cws_l2r, src_cws_r2l)):
            l2r_batch = dy.reshape(dy.concatenate_cols(cws_l2r), (self.embed_size,), batch_size=batch_size)
            l2r_state = l2r_state.add_input(l2r_batch)
            r2l_batch = dy.reshape(dy.concatenate_cols(cws_r2l), (self.embed_size,), batch_size=batch_size)
            r2l_state = r2l_state.add_input(r2l_batch)
            l2r_contexts.append(l2r_state.output())
            r2l_contexts.append(r2l_state.output())
        r2l_contexts.reverse()

        h_fs = []
        for (l2r_i, r2l_i) in zip(l2r_contexts, r2l_contexts):
            h_fs.append(dy.concatenate([l2r_i, r2l_i]))
        h_fs_matrix = dy.concatenate_cols(h_fs)
        fixed_attentional_component = W1_att_f * h_fs_matrix 

        losses = []
        num_words = 0
        
        # Decoder
        tgt_cws = []
        tgt_len = [len(sent) for sent in tgt_batch]
        max_tgt_len = np.max(tgt_len)
        masks = []

        for i in range(max_tgt_len):
            tgt_cws.append([self.tgt_token_to_id[sent[i]] if len(sent) > i else self.tgt_token_to_id['</S>'] for sent in tgt_batch])
            mask = [(1 if len(sent) > i else 0) for sent in tgt_batch]
            masks.append(mask)
            num_words += sum(mask)

        c_t = dy.vecInput(self.hidden_size * 2)
        start_state = dy.affine_transform([b_s, W_s, h_fs[-1]])
        dec_state = self.word_dec_builder.initial_state().set_s([start_state, dy.tanh(start_state)])
        for i, (cws, nws, mask) in enumerate(zip(tgt_cws, tgt_cws[1:], masks)):
            embed_t = dy.lookup_batch(self.tgt_lookup, cws)
            x_t = dy.concatenate([embed_t, c_t])
            dec_state = dec_state.add_input(x_t)
            h_e = dec_state.output()
            c_t = self.__word_attention_mlp(h_fs_matrix, h_e, fixed_attentional_component)
            m_t = dy.tanh(dy.affine_transform([b_m, W_m, dy.concatenate([h_e, c_t])]))
            y_star = dy.affine_transform([b_y, W_y, m_t])
            loss = dy.pickneglogsoftmax_batch(y_star, nws)
            mask_expr = dy.inputVector(mask)
            mask_expr = dy.reshape(mask_expr, (1,),len(batch))
            mask_loss = loss * mask_expr
            losses.append(mask_loss)

        return dy.sum_batches(dy.esum(losses)), num_words

    def train(self, dev, trainer, test, epoch_output=False, output_prefix='translated_test'):
        best_dev_perplexity = 9e9
        for i in range(self.num_epochs):
            total_loss, total_words = 0, 0
            random.shuffle(self.training)
            for j, training_instance in enumerate(self.training):
                loss, num_words = self.__step(training_instance)
                total_loss += loss.scalar_value()
                total_words += num_words
                loss.backward()
                trainer.update()

                if j % 50 == 0:
                    trainer.status()
                    print('Epoch %d sent %d training loss: %f and perplexity: %f' %
                        (i, j, total_loss/total_words, np.exp(total_loss/total_words)))
                    total_loss, total_words = 0, 0

                if j % 3000 == 0:
                    dev_loss, dev_total_words = 0, 0
                    for j, dev_instance in enumerate(dev):
                        loss, num_words = self.__step(dev_instance)
                        dev_loss += loss.scalar_value()
                        dev_total_words += num_words

                    dev_perplexity = np.exp(dev_loss/dev_total_words)
                    print('Epoch %d dev loss: %f and perplexity: %f' % 
                        (i, dev_loss/dev_total_words, dev_perplexity))

                    if dev_perplexity < best_dev_perplexity:
                        best_dev_perplexity = dev_perplexity
                        self.save_model()

            if epoch_output:
                self.translate(test, output_prefix + '_epoch_' + str(i))

    def train_batch(self, dev, trainer, test, epoch_output=False, output_prefix='translated_test'):
        self.training.sort(key=lambda t: len(t[0]), reverse=True)
        dev.sort(key=lambda t: len(t[0]), reverse=True)
        training_order = create_batches(self.training, self.max_batch_size) 
        dev_order = create_batches(dev, self.max_batch_size)
        best_dev_perplexity = 9e9
        for i in range(self.num_epochs):
            total_loss, total_words = 0, 0
            random.shuffle(training_order)
            for j, (start, length) in enumerate(training_order):
                training_batch = self.training[start:start+length]
                loss, num_words = self.__step_batch(training_batch)

                total_loss += loss.scalar_value()
                total_words += num_words
                loss.backward()
                trainer.update()

                if j % 50 == 0:
                    trainer.status()
                    print('Epoch %d batch %d training loss: %f and perplexity: %f' %
                        (i, j, total_loss/total_words, np.exp(total_loss/total_words)))
                    total_loss, total_words = 0, 0

                if j % 3000 == 0:
                    dev_loss, dev_total_words = 0, 0
                    for j, (start, length) in enumerate(dev_order):
                        dev_batch = dev[start:start+length]
                        loss, num_words = self.__step_batch(dev_batch)
                        dev_loss += loss.scalar_value()
                        dev_total_words += num_words

                    dev_perplexity = np.exp(dev_loss/dev_total_words)
                    print('Epoch %d dev loss: %f and perplexity: %f' % 
                        (i, dev_loss/dev_total_words, dev_perplexity))

                    if dev_perplexity < best_dev_perplexity:
                        best_dev_perplexity = dev_perplexity
                        self.save_model()

            if epoch_output:
                self.translate(test, output_prefix + '_epoch_' + str(i))

    def translate(self, src, output_filename):
        outfile = open(trans_out_dir + output_filename, 'wb')
        for sent in src:
            trans_sent = self.translate_sentence(sent)
            outfile.write('%s\n' % trans_sent)
 
def main():
    model = dy.Model()
    trainer = dy.SimpleSGDTrainer(model)
    training_src = read_file(sys.argv[1])
    training_tgt = read_file(sys.argv[2])
    dev_src = read_file(sys.argv[3])
    dev_tgt = read_file(sys.argv[4])
    test_src = read_file(sys.argv[5])
    model_name = sys.argv[6]

    src_vector_file = None
    if len(sys.argv) > 6:
        src_vector_file = sys.argv[7]
    dev = [(x, y) for (x, y) in zip(dev_src, dev_tgt)]
    attention = Attention(model, training_src, training_tgt, model_name,
        src_vectors_file=src_vector_file)

    out_language = sys.argv[1].split('.')[1]

    if LOAD_MODEL:
        attention.load_model()
    if TRAIN:
        attention.train_batch(dev, trainer, test_src, True, 'test.' + out_language)

    attention.translate(test_src, 'test.' + out_language)

if __name__ == '__main__': main()
