import numpy as np
import tensorflow as tf
from tensorflow.contrib import seq2seq
from tensorflow.contrib import training

from utils.helper import ThresholdHelper


class Generator(object):
    def __init__(self, batch_size, seq_len_scr, seq_len_dst, vocab_size_scr, vocab_size_dst, emb_dim, hidden_dim=64,
                 start_token=0, learning_rate=1e-3, name_scope="generator"):
        self.batch_size = batch_size
        self.seq_len_scr = seq_len_scr
        self.seq_len_dst = seq_len_dst
        self.input_scr = tf.placeholder(dtype=tf.int32, shape=[batch_size, seq_len_scr])
        self.input_dst = tf.placeholder(dtype=tf.int32, shape=[batch_size, seq_len_dst])

        self.reward = tf.placeholder(dtype=tf.float32, shape=[batch_size, seq_len_dst])

        start_tokens = (tf.tile([start_token], [batch_size]))

        with tf.variable_scope(name_or_scope=name_scope):
            with tf.variable_scope(name_or_scope="emb"):
                self.encoder_embedding = tf.Variable(tf.truncated_normal(shape=[vocab_size_scr, emb_dim]))
                self.decoder_embedding = tf.Variable(tf.truncated_normal(shape=[vocab_size_dst, emb_dim]))

            encoder_emb_inp = tf.nn.embedding_lookup(self.encoder_embedding, tf.transpose(self.input_scr))

            with tf.variable_scope(name_or_scope="encoder"):
                encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
                encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                    cell=encoder_cell,
                    inputs=encoder_emb_inp,
                    # sequence_length=seq_len_scr,
                    sequence_length=np.tile([seq_len_scr], [batch_size]),
                    time_major=True,
                    dtype=tf.float32
                )

            Wo = tf.Variable(tf.random_normal(shape=[hidden_dim, vocab_size_dst]))
            bo = tf.Variable(tf.random_normal(shape=[vocab_size_dst]))
            output_ids = []
            output_probs = []
            with tf.variable_scope(name_or_scope="decoder"):
                decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
                att_states = tf.transpose(encoder_outputs, [1, 0, 2])

                attention_mechanism = seq2seq.LuongAttention(
                    num_units=hidden_dim,
                    memory=att_states,
                    memory_sequence_length=np.tile([seq_len_scr], [batch_size]),
                )
                decoder_cell = seq2seq.AttentionWrapper(
                    cell=decoder_cell,
                    attention_mechanism=attention_mechanism,
                    attention_layer_size=hidden_dim
                )
                for i in range(seq_len_dst + 1):
                    if i > 0:
                        max_len = i
                        given_ones = self.input_dst[:, :i]
                    else:
                        max_len = seq_len_dst
                        given_ones = self.input_dst
                    threshold = ThresholdHelper(
                        threshold=i,
                        seq_len=max_len,
                        embedding=self.decoder_embedding,
                        given_tokens=given_ones,
                        start_tokens=start_tokens,
                        decision_variables=(Wo, bo)
                    )
                    initial_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32).clone(
                        cell_state=encoder_state)
                    decoder = seq2seq.BasicDecoder(
                        cell=decoder_cell,
                        helper=threshold,
                        initial_state=initial_state
                    )
                    final_outputs, final_state, final_sequence_lengths = seq2seq.dynamic_decode(
                        decoder=decoder,
                        maximum_iterations=max_len
                    )
                    output_ids.append(final_outputs.sample_id)
                    output_probs.append(
                        tf.nn.softmax(
                            tf.tensordot(final_outputs.rnn_output,
                                         Wo,
                                         axes=[[2], [0]]) +
                            bo[None, None, :]))

        self.output_ids = output_ids
        self.output_probs = output_probs

        # pretrain
        def safe_log(x):
            return tf.log(tf.clip_by_value(x, 1e-8, 1 - 1e-8))

        def transform_grads_fn(gradients_to_variables):
            return training.clip_gradient_norms(gradients_to_variables, 5.0)

        logit = safe_log(self.output_probs[seq_len_dst])
        self.log_predictions = logit
        self.prod_mat = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.input_dst, logits=logit), axis=1)
        self.pretrain_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.input_dst, logits=logit))
        self.pretrain_optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate)
        self.pretrain_op = training.create_train_op(self.pretrain_loss, self.pretrain_optimizer,
                                                    transform_grads_fn=transform_grads_fn)
        self.pretrain_summary = tf.summary.scalar(
            "g_pretrain_loss", self.pretrain_loss)

        g_seq = self.output_ids[seq_len_dst]  # follow the generated one
        g_prob = self.output_probs[seq_len_dst]
        g_loss = -tf.reduce_mean(
            tf.reduce_sum(tf.one_hot(g_seq, vocab_size_dst) * safe_log(g_prob), -1) *
            self.reward
        ) / self.seq_len_dst
        g_optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate)
        self.g_op = training.create_train_op(g_loss, g_optimizer, transform_grads_fn=transform_grads_fn)
        self.g_loss = g_loss
        self.g_summary = tf.summary.merge([
            tf.summary.scalar("g_loss", g_loss),
            tf.summary.scalar("g_reward", tf.reduce_mean(self.reward))
        ])

        self.rewards = tf.placeholder(tf.float32, shape=[self.batch_size, self.seq_len_dst,
                                                         vocab_size_dst])  # get from rollout policy and discriminator

        self.cot_loss = -tf.reduce_sum(
            self.output_probs[seq_len_dst] * (self.rewards - self.log_predictions)
        ) / batch_size  # / sequence_length
        cot_optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate)
        self.cot_op = training.create_train_op(self.cot_loss, cot_optimizer, transform_grads_fn=transform_grads_fn)
        self.cot_summary = tf.summary.merge([
            tf.summary.scalar("cot_loss", self.cot_loss),
        ])

    def generate(self, sess, input_scr):
        feed_dict = {
            self.input_scr: input_scr}
        return sess.run(self.output_ids[0], feed_dict=feed_dict)

    def pretrain(self, sess, input_scr, input_dst):
        feed_dict = {
            self.input_scr: input_scr,
            self.input_dst: input_dst}
        _, summary, pretrain_loss = sess.run([self.pretrain_op, self.pretrain_summary, self.pretrain_loss],
                                             feed_dict=feed_dict)
        return pretrain_loss, summary

    def train(self, sess, input_scr, input_dst, rewards):
        feed_dict = {self.input_dst: input_dst,
                     self.input_scr: input_scr,
                     self.reward: rewards}
        _, summary, g_loss = sess.run([self.g_op, self.g_summary, self.g_loss],
                                      feed_dict=feed_dict)
        return g_loss

    def get_loss(self, sess, input_scr, input_dst):
        feed_dict = {self.input_dst: input_dst,
                     self.input_scr: input_scr, }
        summary, g_loss = sess.run([self.pretrain_summary, self.pretrain_loss],
                                   feed_dict=feed_dict)
        return g_loss

    def rollout(self, sess, input_scr, input_dst, keep_steps=0, with_probs=False):
        feed_dict = {self.input_dst: input_dst, self.input_scr: input_scr}
        if with_probs:
            output_tensors = [self.output_ids[keep_steps],
                              self.output_probs[keep_steps]]
        else:
            output_tensors = self.output_ids[keep_steps]
        return sess.run(output_tensors, feed_dict=feed_dict)

    def mc_sample_gen(self, sess, input_scr, input_dst, keep_num):
        # Markov Chain Sample
        mc_sample = self.rollout(
            sess, input_scr, input_dst, keep_steps=keep_num)
        return mc_sample

    def get_reward_med(self, sess, input_scr, input_dst):
        feed_dict = {self.input_dst: input_dst,
                     self.input_scr: input_scr}

        output = sess.run(self.log_predictions, feed_dict=feed_dict)
        return output

    def train_cot(self, sess, input_scr, input_dst, rewards):
        feed_dict = {self.input_dst: input_dst,
                     self.input_scr: input_scr,
                     self.rewards: rewards}
        _, summary, cot_loss = sess.run([self.cot_op, self.cot_summary, self.cot_loss],
                                        feed_dict=feed_dict)
        return cot_loss, summary

    def get_prod_mat(self, sess, input_scr, input_dst):
        feed_dict = {self.input_dst: input_dst, self.input_scr: input_scr}
        prod_mat = sess.run([self.prod_mat], feed_dict=feed_dict)
        return prod_mat[0]