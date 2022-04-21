from flask_restful import Resource, Api
from flask import Flask, request
import tensorflow as tf
from genz_tokenize.preprocess import remove_emoji, convert_unicode, vncore_tokenize, remove_punctuations
from genz_tokenize import Tokenize
from vncorenlp import VnCoreNLP
import numpy as np


label = ['ai', 'cai gi', 'con vat', 'nhu the nao', 'number',
         'tai sao', 'thoi gian', 'time', 'thuc vat', 'yes no', 'location']
postag_label = ['B', 'Np', 'Nc', 'Nu', 'N', 'Ny', 'Ni', 'Nb', 'V', 'Vb', 'A',
                'Ab', 'P', 'R', 'L', 'M', 'E', 'C', 'Cc', 'I', 'T', 'Y', 'Z', 'X', 'CH']

MAXLEN_C = 1000
MAXLEN_Q = 277


class Embedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, dim, max_position_embeddings,  **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.LayerNorm = tf.keras.layers.LayerNormalization(
            epsilon=1e-12, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(rate=0.1)

    def build(self, input_shape: tf.TensorShape):
        with tf.name_scope("word_embeddings"):
            self.weight = self.add_weight(
                name="weight",
                shape=[self.vocab_size, self.dim],
            )

        with tf.name_scope("position_embeddings"):
            self.position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_position_embeddings, self.dim],
            )
        super().build(input_shape)

    def call(self, input_ids=None):
        inputs_embeds = tf.gather(params=self.weight, indices=input_ids)
        input_shape = tf.shape(input_ids)
        position_ids = tf.expand_dims(
            tf.range(start=0, limit=input_shape[-1]), axis=0)
        position_embeds = tf.gather(
            params=self.position_embeddings, indices=position_ids)
        final_embeddings = inputs_embeds + position_embeds
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        final_embeddings = self.dropout(inputs=final_embeddings)
        return final_embeddings


class FeedFoward(tf.keras.layers.Layer):
    def __init__(self, d_model, num_class):
        super(FeedFoward, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.out = tf.keras.layers.Dense(num_class, activation='softmax')

    def call(self, inputs):
        inputs = self.flatten(inputs)
        x = self.fc1(inputs)
        x = self.dropout(x)
        return self.out(x)


class ClassifyModel(tf.keras.Model):
    def __init__(self, vocab_size, max_position_embeddings, d_model, num_heads, num_class):
        super(ClassifyModel, self).__init__()
        self.embedding = Embedding(
            vocab_size, d_model, max_position_embeddings)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads, d_model, dropout=0.1)
        self.fc = tf.keras.layers.Dense(1)
        self.fw = FeedFoward(d_model, num_class)

    def call(self, inputs):
        x = self.embedding(inputs)
        out_mha = self.mha(x, x, x)
        x = self.fc(self.layernorm(x+out_mha))
        return self.fw(x)

    def mask(self, inputs):
        masks = tf.logical_not(tf.math.equal(inputs, 0))
        masks = tf.cast(masks, dtype=tf.float32)
        return masks[:, tf.newaxis, tf.newaxis, :]


def fullConnected(d_model):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(d_model*2, activation='gelu'))
    model.add(tf.keras.layers.Dense(d_model))
    return model


class QuestionAnalys(tf.keras.Model):
    def __init__(self, d_model, maxlen, model_cls, num_heads):
        super(QuestionAnalys, self).__init__()
        self.model_cls = model_cls
        self.model_cls.trainable = False

        self.mha1 = tf.keras.layers.MultiHeadAttention(num_heads, d_model)
        self.mha2 = tf.keras.layers.MultiHeadAttention(num_heads, d_model)

        self.layerNorm1 = tf.keras.layers.LayerNormalization()
        self.layerNorm2 = tf.keras.layers.LayerNormalization()

        self.fc1 = fullConnected(d_model)
        self.fc2 = fullConnected(d_model)
        self.fc3 = fullConnected(d_model)

        self.fc_postag = tf.keras.layers.Dense(d_model)
        self.fc_type = tf.keras.layers.Dense(d_model)

        self.out1 = tf.keras.layers.Dense(1, activation='gelu')
        self.flatten = tf.keras.layers.Flatten()
        self.out2 = tf.keras.layers.Dense(d_model, activation='gelu')
        self.out3 = tf.keras.layers.Dense(maxlen)

        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.dropout2 = tf.keras.layers.Dropout(0.2)

    def mask(self, inputs):
        masks = tf.logical_not(tf.math.equal(inputs, 0))
        masks = tf.cast(masks, dtype=tf.float32)
        return masks[:, tf.newaxis, tf.newaxis, :]

    def call(self, x):
        inputs, input_postag = x
        emb = self.model_cls.embedding(inputs)
        mask = self.mask(inputs)
        out_mha = self.mha1(emb, emb, emb, attention_mask=mask)
        norm = self.layerNorm1(out_mha+emb)
        ff = self.fc1(self.dropout1(norm))

        postag = self.fc_postag(input_postag)
        out_mha = self.mha2(ff, postag, ff, attention_mask=mask)
        norm = self.layerNorm2(ff+out_mha)
        ff = self.fc2(self.dropout2(norm))

        inputs = tf.cast(inputs, dtype=tf.int32)
        out_type = tf.expand_dims(self.fc_type(self.model_cls(inputs)), axis=1)
        out = tf.concat([ff, out_type], axis=1)
        out = self.fc3(out)
        out = self.out1(out)
        out = self.flatten(out)
        out = self.out2(out)
        out = self.out3(out)
        return out


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_size, num_heads, rate):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=hidden_size//num_heads)
        self.layerNorm = tf.keras.layers.LayerNormalization()
        self.fc1 = tf.keras.layers.Dense(hidden_size*4)
        self.fc2 = tf.keras.layers.Dense(hidden_size, activation='gelu')
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, mask, training):
        hidden_state = self.mha(x, x, x, attention_mask=mask)
        hidden_state = self.layerNorm(x+hidden_state)
        hidden_state = self.dropout(hidden_state, training=training)
        hidden_state = self.fc1(hidden_state)
        hidden_state = self.fc2(hidden_state)
        return hidden_state


class Encoder(tf.keras.layers.Layer):
    def __init__(self, hidden_size, num_heads, num_layers, rate):
        super().__init__()
        self.layers = [EncoderLayer(hidden_size, num_heads, rate)
                       for _ in range(num_layers)]

    def call(self, x, mask, training):
        for layer in self.layers:
            x = layer(x, mask, training)
        return x


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_size, num_heads, rate):
        super().__init__()
        self.mha1 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=hidden_size//num_heads)
        self.layerNorm1 = tf.keras.layers.LayerNormalization()
        self.mha2 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=hidden_size//num_heads)
        self.layerNorm2 = tf.keras.layers.LayerNormalization()
        self.fc1 = tf.keras.layers.Dense(hidden_size*4)
        self.fc2 = tf.keras.layers.Dense(hidden_size, activation='gelu')
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, mask, enc_output, training):
        hidden_state = self.mha1(x, x, x, attention_mask=mask)
        hidden_state = self.layerNorm1(x+hidden_state)

        attention_out = self.mha2(hidden_state, enc_output, enc_output)
        hidden_state = self.layerNorm2(attention_out+hidden_state)

        hidden_state = self.dropout(hidden_state, training=training)

        hidden_state = self.fc1(hidden_state)
        hidden_state = self.fc2(hidden_state)

        return hidden_state


class Decoder(tf.keras.layers.Layer):
    def __init__(self, hidden_size, num_heads, num_layers, rate):
        super().__init__()
        self.layers = [DecoderLayer(hidden_size, num_heads, rate)
                       for _ in range(num_layers)]

    def call(self, x, mask, enc_output, training):
        for layer in self.layers:
            x = layer(x, mask, enc_output, training)
        return x


class Rank(tf.keras.Model):
    def __init__(self, model_analys, hidden_size, num_heads, num_layers, rate):
        super().__init__()
        self.model_analys = model_analys
        self.model_analys.trainable = False
        self.hidden_size = hidden_size

        self.encode_question = Encoder(
            hidden_size, num_heads, num_layers, rate)
        self.decode_context = Decoder(hidden_size, num_heads, num_layers, rate)
        self.flatten = tf.keras.layers.Flatten()
        self.out = tf.keras.layers.Dense(1)

    def call(self, document, question, postag, training=False):
        ww = self.model_analys([question, postag])
        ww = tf.expand_dims(ww, axis=-1)
        ww = tf.repeat(ww, self.hidden_size, axis=-1)

        emb_question = self.model_analys.model_cls.embedding(question)*ww
        emb_document = self.model_analys.model_cls.embedding(document)

        q_mask = tf.cast(tf.not_equal(question, 0), dtype=tf.int32)[
            :, tf.newaxis, tf.newaxis, :]
        c_mask = tf.cast(tf.not_equal(document, 0), dtype=tf.int32)[
            :, tf.newaxis, tf.newaxis, :]

        enc_output = self.encode_question(emb_question, q_mask, training)
        dec_output = self.decode_context(
            emb_document, c_mask, enc_output, training)

        out = self.flatten(dec_output)
        return self.out(out)


def getTypeOfWord(vncore, text):
    out = np.zeros(shape=(MAXLEN_Q, len(postag_label)))
    pos = pos_tag(text, vncore)
    for i in range(len(pos)):
        if i == 277:
            print(pos)
        index_pos = postag_label.index(pos[i][1])
        out[i, index_pos] = 1
    return out


def pos_tag(text, vncore):
    text = text.replace('_', ' ')
    tag = vncore.pos_tag(text)
    combine = []
    for i in tag:
        combine += i
    return combine


d_model = 256
num_heads = 6
tokenizer = Tokenize()
model_cls = ClassifyModel(tokenizer.vocab_size(), 1000, 256, 3, len(label))
model_analys = QuestionAnalys(d_model, MAXLEN_Q, model_cls, num_heads)
model = Rank(model_analys, hidden_size=256,
             num_heads=4, num_layers=4, rate=0.05)
ckpt = tf.train.Checkpoint(model=model)

ckpt_manager = tf.train.CheckpointManager(ckpt, 'checkpoint', max_to_keep=2)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

corenlp = VnCoreNLP(address="http://127.0.0.1", port=9000)
app = Flask(__name__)
api = Api(app)


class Rank(Resource):
    def __init__(self) -> None:
        super().__init__()

    def post(self):
        document = request.form['document']
        question = request.form['question']
        question = remove_punctuations(remove_emoji(
            vncore_tokenize(convert_unicode(question), corenlp)))
        document = document.split('\n')
        document_process = [
            remove_punctuations(
                remove_emoji(
                    vncore_tokenize(
                        convert_unicode(i),
                        corenlp
                    )
                )
            )
            for i in document
        ]
        input_question = []
        input_context = []
        input_postag = []
        q = tokenizer(question, max_len=MAXLEN_Q,
                      padding=True, truncation=True)
        for i in document_process:
            c = tokenizer(i, max_len=MAXLEN_C,
                          padding=True, truncation=True)
            input_question.append(q['input_ids'])
            input_context.append(c['input_ids'])
            input_postag.append(getTypeOfWord(corenlp, question).tolist())
        input_question = np.array(input_question, dtype=np.int32)
        input_context = np.array(input_context, dtype=np.int32)
        input_postag = np.array(input_postag, dtype=np.int32)

        predicts = tf.nn.sigmoid(
            model(input_context, input_question, input_postag)).numpy()
        index = np.argmax(predicts[:, 0])
        return {'result': document[index]}


api.add_resource(Rank, '/')
if __name__ == '__main__':
    app.run(debug=True, port=8081)
