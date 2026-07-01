"""Tacotron2-lite — a from-scratch seq2seq TTS baseline in Keras/TensorFlow (Part A1).

Encoder (Embedding -> Conv stack -> BiLSTM) -> Bahdanau-attention LSTM decoder
(autoregressive, teacher-forced at train time) -> Postnet. This is the deliberately weak
baseline: on 1-5 min of data it is expected to show attention-collapse / exposure-bias
failures, which we analyse before moving to XTTS. Course anchors: ML4.pptx (RNN/LSTM),
attention.pptx (attention).

NOTE: heavy custom code, validated on the server — expect to iterate on the first run.
"""
from __future__ import annotations

from src.creation.keras_tts.audio_tts import CFG
from src.creation.keras_tts.text import VOCAB_SIZE


def build_model(n_mels: int = CFG.n_mels, enc_dim: int = 256, dec_units: int = 512,
                attn_dim: int = 128):
    import tensorflow as tf
    import keras
    from keras import layers

    class AttnLSTMCell(layers.Layer):
        """LSTM decoder cell with Bahdanau attention over the encoder memory.

        Memory is set on the cell by the parent model before each call (constant across
        decoder steps), avoiding the RNN `constants` API.
        """

        def __init__(self, units, n_mels, attn_dim, **kw):
            super().__init__(**kw)
            self.units, self.n_mels = units, n_mels
            self.state_size = [units, units]
            self.output_size = n_mels + 1
            self.pre1 = layers.Dense(256, activation="relu")
            self.pre2 = layers.Dense(256, activation="relu")
            self.drop = layers.Dropout(0.5)
            self.lstm = layers.LSTMCell(units)
            self.w_query = layers.Dense(attn_dim)
            self.v = layers.Dense(1)
            self.mel_proj = layers.Dense(n_mels)
            self.stop_proj = layers.Dense(1)
            self._memory = None        # (B, Te, M)   set by parent
            self._proc_memory = None   # (B, Te, attn) set by parent

        def call(self, inputs, states, training=None):
            h, c = states
            # prenet with always-on dropout (Tacotron trick that aids generalization)
            p = self.drop(self.pre1(inputs), training=True)
            p = self.drop(self.pre2(p), training=True)
            query = tf.expand_dims(self.w_query(h), 1)            # (B,1,attn)
            score = self.v(tf.tanh(self._proc_memory + query))    # (B,Te,1)
            align = tf.nn.softmax(score, axis=1)
            context = tf.reduce_sum(align * self._memory, axis=1)  # (B, M)
            lstm_in = tf.concat([p, context], axis=-1)
            _, [h2, c2] = self.lstm(lstm_in, [h, c])
            proj_in = tf.concat([h2, context], axis=-1)
            out = tf.concat([self.mel_proj(proj_in), self.stop_proj(proj_in)], axis=-1)
            return out, [h2, c2]

    def _encoder():
        ids = keras.Input(shape=(None,), dtype="int32", name="text")
        x = layers.Embedding(VOCAB_SIZE, enc_dim, mask_zero=False)(ids)
        for _ in range(3):
            x = layers.Conv1D(enc_dim, 5, padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.Dropout(0.5)(x)
        x = layers.Bidirectional(layers.LSTM(enc_dim // 2, return_sequences=True))(x)
        return keras.Model(ids, x, name="encoder")

    def _postnet():
        m = keras.Input(shape=(None, n_mels))
        x = m
        for _ in range(4):
            x = layers.Conv1D(256, 5, padding="same", activation="tanh")(x)
            x = layers.BatchNormalization()(x)
        x = layers.Conv1D(n_mels, 5, padding="same")(x)
        return keras.Model(m, x, name="postnet")

    class Tacotron2Lite(keras.Model):
        def __init__(self):
            super().__init__()
            self.encoder = _encoder()
            self.mem_dense = layers.Dense(attn_dim, name="memory_proj")
            self.cell = AttnLSTMCell(dec_units, n_mels, attn_dim)
            self.dec_rnn = layers.RNN(self.cell, return_sequences=True)
            self.postnet = _postnet()
            self.n_mels = n_mels

        def call(self, inputs, training=None):
            text_ids, dec_inputs = inputs
            memory = self.encoder(text_ids, training=training)
            self.cell._memory = memory
            self.cell._proc_memory = self.mem_dense(memory)
            y = self.dec_rnn(dec_inputs, training=training)     # (B, Tm, n_mels+1)
            mel_pre = y[..., : self.n_mels]
            stop = y[..., self.n_mels:]
            mel_post = mel_pre + self.postnet(mel_pre, training=training)
            return mel_pre, mel_post, stop

        def train_step(self, batch):
            text_ids, mel, mel_len = batch
            tm = tf.shape(mel)[1]
            go = tf.zeros_like(mel[:, :1, :])
            dec_inputs = tf.concat([go, mel[:, :-1, :]], axis=1)   # teacher forcing
            mask = tf.sequence_mask(mel_len, tm, dtype=tf.float32)  # (B, Tm)
            stop_tgt = tf.one_hot(mel_len - 1, tm)                  # 1 at final frame

            with tf.GradientTape() as tape:
                mel_pre, mel_post, stop = self((text_ids, dec_inputs), training=True)
                m = mask[..., None]
                denom = tf.reduce_sum(mask) * self.n_mels + 1e-6
                l_pre = tf.reduce_sum(tf.abs(mel_pre - mel) * m) / denom
                l_post = tf.reduce_sum(tf.abs(mel_post - mel) * m) / denom
                bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=stop_tgt, logits=stop[..., 0])
                l_stop = tf.reduce_sum(bce * mask) / (tf.reduce_sum(mask) + 1e-6)
                loss = l_pre + l_post + l_stop
            self.optimizer.apply_gradients(zip(tape.gradient(loss, self.trainable_variables),
                                               self.trainable_variables))
            return {"loss": loss, "mel": l_pre + l_post, "stop": l_stop}

    return Tacotron2Lite()
