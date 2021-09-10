from tensorflow.contrib.seq2seq import Helper
import tensorflow as tf


class ThresholdHelper(Helper):
    """
        This helper can:
            1. Translate discrete tokens into embeddings
            2. Take tokens from the given sequence before a threshold T,
            and sample tokens after T so that:
                pretrain setting equals to T==seq_len
                generate setting equals to T==0
                rollout setting equals to T==given_len
            3. Add a decision layer after RNN
    """

    def __init__(
            self,
            threshold,
            seq_len,
            embedding,
            given_tokens,  # known input
            start_tokens,  # unknown input
            decision_variables,  # (W, b)
            input_scr_onehot,
            encoder_state,
            ptn_paras,
    ):
        self._threshold = threshold
        self._seq_len = seq_len
        self._batch_size = tf.size(start_tokens)

        if callable(embedding):
            self._embedding_fn = embedding
        else:
            self._embedding_fn = (
                lambda ids: tf.nn.embedding_lookup(embedding, ids))

        self._given_tokens = tf.convert_to_tensor(given_tokens, dtype=tf.int32, name="given_input")
        self._given_emb = self._embedding_fn(self._given_tokens)

        self._start_tokens = tf.convert_to_tensor(
            start_tokens, dtype=tf.int32, name="start_tokens")
        self._start_emb = self._embedding_fn(self._start_tokens)

        self._decision_W, self._decision_b = decision_variables
        self.input_scr_onehot = input_scr_onehot
        self.encoder_state = encoder_state
        (self.Wh, self.Ws, self.Wx, self.b_ptr) = ptn_paras

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return tf.int32

    def initialize(self, name=None):
        """Returns `(initial_finished, initial_inputs)`."""
        finished = tf.tile([False], [self._batch_size])
        return finished, self._start_emb

    def sample(self, time, outputs, state, name=None):
        return None

    def sample_ptn(self, time, outputs, state, cur_inputs, get_prob=False, name=None):
        """Returns `sample_ids`."""
        attn_dist = state.alignments
        prob_attn = tf.expand_dims(attn_dist,
                                   2)
        prob_attn = tf.matmul(a=self.input_scr_onehot, b=prob_attn)
        prob_attn = tf.squeeze(prob_attn)

        prob_lstm = tf.nn.softmax(
            tf.matmul(outputs, self._decision_W) + self._decision_b)
        state_s = state.cell_state.h
        p_gen = tf.matmul(self.encoder_state.h, self.Wh) + tf.matmul(state_s, self.Ws) + \
                tf.matmul(cur_inputs, self.Wx) + self.b_ptr
        p_gen = tf.sigmoid(p_gen)
        # p_gen = tf.squeeze(p_gen)
        # prob = tf.transpose(p_gen) * tf.transpose(prob_lstm) + tf.transpose(1 - p_gen) * tf.transpose(prob_attn)
        # prob = tf.transpose(prob)
        prob = p_gen * prob_lstm + (1 - p_gen) * prob_attn
        log_prob = tf.log(prob)
        sample_ids = tf.cast(
            tf.reshape(
                tf.multinomial(log_prob, 1),
                [self._batch_size]
            ), tf.int32)
        # free generation
        if self._threshold <= 0:
            if get_prob:
                return sample_ids, state_s
            return sample_ids
        # teacher forcing (time < threshold) + free(o.w.)
        else:
            finished = tf.greater_equal(time, self._seq_len)
            sample_ids = tf.cond(
                tf.logical_or(tf.greater_equal(time, self._threshold),
                              finished),
                lambda: sample_ids,
                lambda: self._given_tokens[:, time],
            )
        if get_prob:
            return sample_ids, state_s
        return sample_ids

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        """Returns `(finished, next_inputs, next_state)`."""
        next_time = time + 1
        finished = tf.greater_equal(next_time, self._seq_len)
        next_inputs = self._embedding_fn(sample_ids)
        # if self._threshold <= 0:
        #     next_inputs = self._embedding_fn(sample_ids)
        # else:
        #     next_inputs = tf.cond(
        #         tf.logical_or(tf.greater_equal(next_time, self._threshold),
        #                       finished),
        #         lambda: self._embedding_fn(sample_ids),
        #         lambda: self._given_emb[:, time, :],
        #     )
        return (finished, next_inputs, state)
