import nltk
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn import preprocessing

import utils.text_preprocess as tp
from generator_t2t import Generator

tf.flags.DEFINE_string("training_data_path", "../data/webnlg_new.txt", "")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (Default: 64)")
tf.flags.DEFINE_integer("num_iter", 1, "Number of training epochs (Default: 100)")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_integer("text_embedding_dim", 300, "Dimensionality of word embedding (Default: 300)")
tf.flags.DEFINE_integer("position_embedding_dim", 100, "Dimensionality of position embedding (Default: 100)")
tf.flags.DEFINE_string("filter_sizes", "2,3,4,5", "Comma-separated filter sizes (Default: 2,3,4,5)")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (Default: 128)")
tf.flags.DEFINE_float("l2_reg_lambda", 3.0, "L2 regularization lambda (Default: 3.0)")

tf.flags.DEFINE_integer("rollout_num", 8, "roll out number")
tf.flags.DEFINE_integer("g_pretrain", 1, "pretain of g")
tf.flags.DEFINE_integer("t2t_step", 100, "g step")
tf.flags.DEFINE_integer("m_step", 1, "m step")
tf.flags.DEFINE_integer("g_step", 1, "g step")
# tf.flags.DEFINE_integer("max_batch", 6, "g step")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = data_size // batch_size

    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def chinese_tokenize(s):
    new_s = ""
    cur_state = False
    for uchar in s:
        if uchar >= u'\u4e00' and uchar <= u'\u9fa5' or (uchar in "[$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+"):
            if cur_state:
                new_s+=" "
            new_s+=(uchar + ' ')
        else:
            cur_state = True
            new_s += uchar
    return new_s

# def tokenlizer(_iter):
#     for value in _iter:
#         yield nltk.word_tokenize(value)

def tokenlizer(_iter):
    for value in _iter:
        yield nltk.word_tokenize(value)

def sigmoid(x): return 1 / (1 + np.exp(-x))


def save_output(gen_x, gen_y, replace_rules, generator, sess,
                word_vocab_processor, tuple_vocab_processor, file_name):
    import os
    if os.path.isfile(file_name):
        os.remove(file_name)
    gen_batches_test = batch_iter(list(zip(gen_x, gen_y, replace_rules)),
                                  FLAGS.batch_size, FLAGS.num_iter, shuffle=False)
    for g_batch in gen_batches_test:
        g_x, g_y, re_rule = zip(*g_batch)
        generated_sentences = list(generator.generate(sess, g_x))
        generated_sentences = list(word_vocab_processor.reverse(generated_sentences))
        tuple_ = list(tuple_vocab_processor.reverse(g_x))
        with open("delex" + file_name, "a", encoding="UTF-8") as outfile:
            for tuple_sent, sent in zip(tuple_, generated_sentences):
                outfile.write(tuple_sent.replace("<UNK>", " ") + "\n")
                outfile.write(sent.replace("<UNK>", " ") + "\n")
        with open(file_name, "a", encoding="UTF-8") as outfile:
            tuple_ = tp.relex_list(tuple_, re_rule)
            generated_sentences = tp.relex_list(generated_sentences, re_rule)
            for tuple_sent, sent in zip(tuple_, generated_sentences):
                outfile.write(tuple_sent.replace("<UNK>", " ") + "\n")
                outfile.write(sent.replace("<UNK>", " ") + "\n")



def train():

    # the number of test set
    dev_num = 2048

    original_text, delex_text, original_tuples, delex_tuples, replace_rules = \
        tp.init_data(data_path=FLAGS.training_data_path)
    sentences = original_text
    delex_relation_sentences = original_tuples
    max_seq_len = tp.get_max_len(sentences)
    dev_index = len(sentences) - dev_num

    max_relation_len = tp.get_max_len(delex_tuples)
    word_vocab_processor = preprocessing.VocabularyProcessor(max_seq_len, tokenizer_fn=tokenlizer)
    tuple_vocab_processor = preprocessing.VocabularyProcessor(max_relation_len, tokenizer_fn=tokenlizer)
    word_vocab_processor.fit(sentences)
    tuple_vocab_processor.fit(delex_relation_sentences)
    text_vec = np.array(list(word_vocab_processor.transform(sentences)))
    tuple_vec = np.array(list(tuple_vocab_processor.transform(delex_relation_sentences)))
    gen_x = tuple_vec
    gen_y = text_vec
    log_train_loss = open("logs/train_loss", "w")
    log_nll_g = open("logs/nll_g", "w")
    log_nll_m = open("logs/nll_m", "w")

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        writer = tf.summary.FileWriter("logs/", sess.graph)

        generator = Generator(
            batch_size=FLAGS.batch_size,
            seq_len_scr=max_relation_len,
            seq_len_dst=max_seq_len,
            vocab_size_scr=len(tuple_vocab_processor.vocabulary_),
            vocab_size_dst=len(word_vocab_processor.vocabulary_),
            emb_dim=500,
            hidden_dim=500,
            name_scope="gen"
        )
        med = Generator(
            batch_size=FLAGS.batch_size,
            seq_len_scr=max_relation_len,
            seq_len_dst=max_seq_len,
            vocab_size_scr=len(tuple_vocab_processor.vocabulary_),
            vocab_size_dst=len(word_vocab_processor.vocabulary_),
            emb_dim=500,
            hidden_dim=500,
            name_scope="med"
        )
        writer_step = 0
        global_step = tf.Variable(0, name='global_step', trainable=False)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        # pre-train G
        for i in range(FLAGS.g_pretrain):
            loss_all = []
            gen_batches = batch_iter(list(zip(gen_x[:dev_index], gen_y[:dev_index])), FLAGS.batch_size, FLAGS.num_iter,
                                     shuffle=True)
            for g_batch in gen_batches:
                g_x, g_y = zip(*g_batch)
                loss, summary = generator.pretrain(sess, input_scr=g_x, input_dst=g_y)
                writer.add_summary(summary, global_step=writer_step)
                writer_step += 1
                loss_all.append(loss)
            print("epoch " + str(i) + " finished!")
            print("loss" + str(sum(loss_all) / len(loss_all)))
            log_train_loss.write("epoch " + str(i) + " finished!" + "\n")
            log_train_loss.write("loss" + str(sum(loss_all) / len(loss_all)) + "\n")
            log_train_loss.flush()
            file_name = "res_pre" + str(i) + ".txt"
            save_output(gen_x, gen_y, replace_rules, generator, sess, word_vocab_processor, tuple_vocab_processor,
                        file_name)
            global_step = tf.assign_add(global_step, 1, name="global_step")
            saver.save(sess, "./save/", global_step=global_step)

        writer_step = 0
        for i in range(FLAGS.t2t_step):
            merged_y = np.array(gen_y[:dev_index])
            loss_all = []
            med_batches = batch_iter(list(zip(gen_x[:dev_index], merged_y)), FLAGS.batch_size, FLAGS.num_iter)
            for m_batch in med_batches:
                m_x, m_y = zip(*m_batch)
                loss, summary = med.pretrain(sess, input_scr=m_x, input_dst=m_y)
                writer.add_summary(summary, global_step=writer_step)
                writer_step += 1
                loss_all.append(loss)
            print("epoch " + str(i) + " finished!")
            print("loss: " + str(sum(loss_all) / len(loss_all)))
            log_train_loss.write("epoch " + str(i) + " finished!" + "\n")
            log_train_loss.write("G_loss on cot: " + str(sum(loss_all) / len(loss_all)) + "\n")
            log_train_loss.flush()
            # train G
            gen_batches_pretrain = batch_iter(list(zip(gen_x[:dev_index], gen_y[:dev_index])), FLAGS.batch_size,
                                              FLAGS.num_iter)
            generated_sentences_list = []

            for g_batch in gen_batches_pretrain:
                g_x, g_y = zip(*g_batch)
                generated_sentences = list(generator.generate(sess, g_x))
                # generated_sentences = list(word_vocab_processor.reverse(generated_sentences))
                generated_sentences_list += generated_sentences

            g_batches = batch_iter(list(zip(gen_x[:dev_index], generated_sentences_list)), FLAGS.batch_size,
                                   FLAGS.num_iter)
            loss_all = []

            for g_batch in g_batches:
                g_x, g_y = zip(*g_batch)
                reward = med.get_reward_med(sess, input_scr=g_x, input_dst=g_y)
                cot_loss, summary = generator.train_cot(sess, g_x, g_y, reward)
                writer.add_summary(summary, global_step=writer_step)
                writer_step += 1
                loss_all.append(cot_loss)
            print("epoch " + str(i) + " finished!")
            print("G_loss on cot: " + str(sum(loss_all) / len(loss_all)))
            log_train_loss.write("epoch " + str(i) + " finished!" + "\n")
            log_train_loss.write("G_loss on cot: " + str(sum(loss_all) / len(loss_all)) + "\n")
            log_train_loss.flush()

            # eval G
            cot_loss_g = []
            cot_loss_m = []
            g_batches_test = batch_iter(list(zip(gen_x[dev_index:], gen_y[dev_index:])), FLAGS.batch_size,
                                        FLAGS.num_iter)
            for g_batch in g_batches_test:
                g_x, g_y = zip(*g_batch)
                loss = generator.get_loss(sess, g_x, g_y, )
                cot_loss_g.append(loss)
                loss = med.get_loss(sess, g_x, g_y, )
                cot_loss_m.append(loss)
            print("NLL_test_g: " + str(sum(cot_loss_g) / len(cot_loss_g)))
            print("NLL_test_m: " + str(sum(cot_loss_m) / len(cot_loss_m)))
            log_nll_g.write("epoch " + str(i) + " finished!" + "\n")
            log_nll_g.write("NLL_test_g: " + str(sum(cot_loss_g) / len(cot_loss_g)) + "\n")
            log_nll_m.write("epoch " + str(i) + " finished!" + "\n")
            log_nll_m.write("NLL_test_m: " + str(sum(cot_loss_m) / len(cot_loss_m)) + "\n")

            file_name = "res_cot" + str(i) + ".txt"
            save_output(gen_x, gen_y, replace_rules, generator, sess, word_vocab_processor, tuple_vocab_processor,
                        file_name)
            global_step = tf.assign_add(global_step, 1, name="global_step")
            saver.save(sess, "./save/", global_step=global_step)
    log_train_loss.close()
    log_nll_g.close()
    log_nll_m.close()


if __name__ == '__main__':
    train()
