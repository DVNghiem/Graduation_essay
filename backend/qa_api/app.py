import tensorflow as tf
from genz_tokenize import TokenizeForBert
from genz_tokenize.preprocess import remove_emoji, convert_unicode, vncore_tokenize
import numpy as np
import math
from typing import Dict, List, Optional, Union, Tuple
from flask import Flask, request
from flask_restful import Resource, Api
from vncorenlp import VnCoreNLP

# ===============================================


def get_initializer(initializer_range: float = 0.02) -> tf.initializers.TruncatedNormal:
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)


def shape_list(tensor: Union[tf.Tensor, np.ndarray]) -> List[int]:
    if isinstance(tensor, np.ndarray):
        return list(tensor.shape)

    dynamic = tf.shape(tensor)

    if tensor.shape == tf.TensorShape(None):
        return dynamic

    static = tensor.shape.as_list()

    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


class Config:
    def __init__(
        self,
        input_vocab_size=50265,
        target_vocab_size=40000,
        max_position_embeddings=258,
        hidden_size=256,
        initializer_range=0.02,
        layer_norm_eps=1e-6,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        num_attention_heads=8,
        num_hidden_layers=8,
        is_decoder=True,
        intermediate_size=1024,
        type_vocab_size=1
    ):
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.is_decoder = is_decoder
        self.intermediate_size = intermediate_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.type_vocab_size = type_vocab_size


class TFRobertaEmbeddings(tf.keras.layers.Layer):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config, vocab_size, **kwargs):
        super().__init__(**kwargs)

        self.padding_idx = 0
        self.vocab_size = vocab_size
        self.type_vocab_size = config.type_vocab_size
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings
        self.initializer_range = config.initializer_range
        self.LayerNorm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def build(self, input_shape: tf.TensorShape):
        with tf.name_scope("word_embeddings"):
            self.weight = self.add_weight(
                name="weight",
                shape=[self.vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        with tf.name_scope("token_type_embeddings"):
            self.token_type_embeddings = self.add_weight(
                name="embeddings_token",
                shape=[self.type_vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        with tf.name_scope("position_embeddings"):
            self.position_embeddings = self.add_weight(
                name="embeddings_position",
                shape=[self.max_position_embeddings, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        super().build(input_shape)

    def create_position_ids_from_input_ids(self, input_ids, past_key_values_length=0):
        mask = tf.cast(tf.math.not_equal(
            input_ids, self.padding_idx), dtype=input_ids.dtype)
        incremental_indices = (tf.math.cumsum(
            mask, axis=1) + past_key_values_length) * mask

        return incremental_indices + self.padding_idx

    def call(
        self,
        input_ids=None,
        position_ids=None,
        token_type_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
        training=False,
    ):
        """
        Applies embedding based on inputs tensor.
        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        assert not (input_ids is None and inputs_embeds is None)

        if input_ids is not None:
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        input_shape = shape_list(inputs_embeds)[:-1]

        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)

        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = self.create_position_ids_from_input_ids(
                    input_ids=input_ids, past_key_values_length=past_key_values_length
                )
            else:
                position_ids = tf.expand_dims(
                    tf.range(start=self.padding_idx + 1, limit=input_shape[-1] + self.padding_idx + 1), axis=0
                )

        position_embeds = tf.gather(
            params=self.position_embeddings, indices=position_ids)
        token_type_embeds = tf.gather(
            params=self.token_type_embeddings, indices=token_type_ids)
        final_embeddings = inputs_embeds + position_embeds + token_type_embeds
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        final_embeddings = self.dropout(
            inputs=final_embeddings, training=training)

        return final_embeddings


class TFRobertaSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

        self.query = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        self.key = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key"
        )
        self.value = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )
        self.dropout = tf.keras.layers.Dropout(
            rate=config.attention_probs_dropout_prob)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        # Reshape from [batch_size, seq_length, all_head_size] to [batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(tensor=tensor, shape=(
            batch_size, -1, self.num_attention_heads, self.attention_head_size))

        # Transpose the tensor from [batch_size, seq_length, num_attention_heads, attention_head_size] to [batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        encoder_hidden_states: tf.Tensor,
        encoder_attention_mask: tf.Tensor,
        past_key_value: Tuple[tf.Tensor],
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        batch_size = shape_list(hidden_states)[0]
        mixed_query_layer = self.query(inputs=hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(
                self.key(inputs=encoder_hidden_states), batch_size)
            value_layer = self.transpose_for_scores(
                self.value(inputs=encoder_hidden_states), batch_size)
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(
                self.key(inputs=hidden_states), batch_size)
            value_layer = self.transpose_for_scores(
                self.value(inputs=hidden_states), batch_size)
            key_layer = tf.concat([past_key_value[0], key_layer], axis=2)
            value_layer = tf.concat([past_key_value[1], value_layer], axis=2)
        else:
            key_layer = self.transpose_for_scores(
                self.key(inputs=hidden_states), batch_size)
            value_layer = self.transpose_for_scores(
                self.value(inputs=hidden_states), batch_size)

        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)

        if self.is_decoder:
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # (batch size, num_heads, seq_len_q, seq_len_k)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(attention_scores, dk)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in TFRobertaModel call() function)
            attention_scores = tf.add(attention_scores, attention_mask)

        # Normalize the attention scores to probabilities.
        attention_probs = tf.nn.softmax(logits=attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(
            inputs=attention_probs, training=training)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = tf.multiply(attention_probs, head_mask)

        attention_output = tf.matmul(attention_probs, value_layer)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, all_head_size)
        attention_output = tf.reshape(
            tensor=attention_output, shape=(batch_size, -1, self.all_head_size))
        outputs = (attention_output, attention_probs) if output_attentions else (
            attention_output,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs  # tuple


class TFRobertaSelfOutput(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.LayerNorm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states


class TFRobertaAttention(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.self_attention = TFRobertaSelfAttention(config, name="self")
        self.dense_output = TFRobertaSelfOutput(config, name="output")

    def call(
        self,
        input_tensor: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        encoder_hidden_states: tf.Tensor,
        encoder_attention_mask: tf.Tensor,
        past_key_value: Tuple[tf.Tensor],
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        self_outputs = self.self_attention(
            hidden_states=input_tensor,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            training=training,
        )
        attention_output = self.dense_output(
            hidden_states=self_outputs[0], input_tensor=input_tensor, training=training
        )
        # add attentions (possibly with past_key_value) if we output them
        outputs = (attention_output,) + self_outputs[1:]

        return outputs


class TFRobertaIntermediate(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.intermediate_act_fn = tf.nn.gelu

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class TFRobertaOutput(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.LayerNorm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states


class TFRobertaEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.attention = TFRobertaAttention(config, name="attention")
        self.intermediate = TFRobertaIntermediate(config, name="intermediate")
        self.bert_output = TFRobertaOutput(config, name="output")

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        past_key_value: Optional[Tuple[tf.Tensor]],
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:
                                                  2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            input_tensor=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=self_attn_past_key_value,
            output_attentions=output_attentions,
            training=training,
        )
        attention_output = self_attention_outputs[0]

        # add self attentions if we output attention weights
        outputs = self_attention_outputs[1:]

        intermediate_output = self.intermediate(hidden_states=attention_output)
        layer_output = self.bert_output(
            hidden_states=intermediate_output, input_tensor=attention_output, training=training
        )
        outputs = (layer_output,) + outputs  # add attentions if we output them
        return outputs


class TFRobertaEncoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.layer = [TFRobertaEncoderLayer(
            config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        past_key_values: Optional[Tuple[Tuple[tf.Tensor]]],
        training: bool = False,
    ) -> tf.Tensor:
        attention_mask_shape = shape_list(attention_mask)
        extended_attention_mask = tf.reshape(
            attention_mask, (attention_mask_shape[0],
                             1, 1, attention_mask_shape[1])
        )
        extended_attention_mask = tf.cast(
            extended_attention_mask, dtype=hidden_states.dtype)
        one_cst = tf.constant(1.0, dtype=hidden_states.dtype)
        ten_thousand_cst = tf.constant(-10000.0, dtype=hidden_states.dtype)
        extended_attention_mask = tf.multiply(tf.subtract(
            one_cst, extended_attention_mask), ten_thousand_cst)

        for i, layer_module in enumerate(self.layer):

            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=extended_attention_mask,
                head_mask=head_mask[i],
                past_key_value=past_key_value,
                output_attentions=True,
                training=training,
            )
            hidden_states = layer_outputs[0]

        return hidden_states


class TFRobertaDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.attention = TFRobertaAttention(config, name="attention")

        self.crossattention = TFRobertaAttention(config, name="crossattention")
        self.intermediate = TFRobertaIntermediate(config, name="intermediate")
        self.bert_output = TFRobertaOutput(config, name="output")

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        encoder_hidden_states: Optional[tf.Tensor],
        encoder_attention_mask: Optional[tf.Tensor],
        past_key_value: Optional[Tuple[tf.Tensor]],
        output_attentions: bool,
        training: bool = False,
    ) -> tf.Tensor:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:
                                                  2] if past_key_value is not None else None

        self_attention_outputs = self.attention(
            input_tensor=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=self_attn_past_key_value,
            output_attentions=output_attentions,
            training=training,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        outputs = self_attention_outputs[1:-1]
        present_key_value = self_attention_outputs[-1]

        cross_attn_present_key_value = None
        # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
        cross_attn_past_key_value = past_key_value[-2:
                                                   ] if past_key_value is not None else None
        cross_attention_outputs = self.crossattention(
            input_tensor=attention_output,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=cross_attn_past_key_value,
            output_attentions=output_attentions,
            training=training,
        )
        attention_output = cross_attention_outputs[0]
        # add cross attentions if we output attention weights
        outputs = outputs + cross_attention_outputs[1:-1]

        # add cross-attn cache to positions 3,4 of present_key_value tuple
        cross_attn_present_key_value = cross_attention_outputs[-1]
        present_key_value = present_key_value + cross_attn_present_key_value

        intermediate_output = self.intermediate(hidden_states=attention_output)
        layer_output = self.bert_output(
            hidden_states=intermediate_output, input_tensor=attention_output, training=training
        )
        outputs = (layer_output,) + outputs  # add attentions if we output them

        # if decoder, return the attn key/values as the last output
        outputs = outputs + (present_key_value,)
        return outputs


class TFRobertaDecoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.layer = [TFRobertaDecoderLayer(
            config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        encoder_hidden_states: Optional[tf.Tensor],
        encoder_attention_mask: Optional[tf.Tensor],
        past_key_values: Optional[Tuple[Tuple[tf.Tensor]]],
        training: bool = False,
    ) -> tf.Tensor:
        attention_mask_shape = shape_list(attention_mask)
        input_shape = shape_list(hidden_states)
        batch_size, seq_length, _ = input_shape
        mask_seq_length = seq_length
        seq_ids = tf.range(mask_seq_length)

        causal_mask = tf.less_equal(
            tf.tile(seq_ids[None, None, :], (batch_size, mask_seq_length, 1)),
            seq_ids[None, :, None],
        )

        causal_mask = tf.cast(causal_mask, dtype=attention_mask.dtype)
        extended_attention_mask = causal_mask * attention_mask[:, None, :]
        attention_mask_shape = shape_list(extended_attention_mask)
        extended_attention_mask = tf.reshape(
            extended_attention_mask, (
                attention_mask_shape[0], 1, attention_mask_shape[1], attention_mask_shape[2])
        )

        extended_attention_mask = tf.cast(
            extended_attention_mask, dtype=hidden_states.dtype)
        one_cst = tf.constant(1.0, dtype=hidden_states.dtype)
        ten_thousand_cst = tf.constant(-10000.0, dtype=hidden_states.dtype)
        extended_attention_mask = tf.multiply(tf.subtract(
            one_cst, extended_attention_mask), ten_thousand_cst)

        encoder_attention_mask = tf.cast(
            encoder_attention_mask, dtype=extended_attention_mask.dtype)
        encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

        encoder_extended_attention_mask = (
            1.0 - encoder_extended_attention_mask) * -10000.0

        for i, layer_module in enumerate(self.layer):

            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=extended_attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                past_key_value=past_key_value,
                output_attentions=True,
                training=training,
            )
            hidden_states = layer_outputs[0]
        return hidden_states


class TFRobertaModel(tf.keras.Model):
    def __init__(self, config, maxlen_c, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        self.embedding_encoder = TFRobertaEmbeddings(
            config, config.input_vocab_size, name='embedding_encoder')
        self.embedding_decoder = TFRobertaEmbeddings(
            config, config.target_vocab_size, name='embedding_decoder')
        self.encoder = TFRobertaEncoder(config, name='encoder')
        self.decoder = TFRobertaDecoder(config, name='decoder')

        self.logit = tf.keras.layers.Dense(2)
        self.start = tf.keras.layers.Dense(maxlen_c)
        self.end = tf.keras.layers.Dense(maxlen_c)

    def call(self,
             encoder_inputs: Optional[tf.Tensor],
             encoder_attention_mask: Optional[tf.Tensor],
             encoder_token_type_id: tf.Tensor,
             head_mask: tf.Tensor,
             decoder_inputs: tf.Tensor,
             decoder_attention_mask: tf.Tensor,
             decoder_token_type_id: tf.Tensor,
             training: bool = False):
        if head_mask is None:
            head_mask = [None] * self.config.num_hidden_layers
        if encoder_token_type_id is None:
            input_shape = shape_list(encoder_inputs)
            encoder_token_type_id = tf.fill(dims=input_shape, value=0)
        hidden_states = self.embedding_encoder(
            input_ids=encoder_inputs, token_type_ids=encoder_token_type_id)
        encoder_output = self.encoder(
            hidden_states=hidden_states,
            attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            past_key_values=None,
            training=training
        )

        if decoder_token_type_id is None:
            input_shape = shape_list(decoder_inputs)
            decoder_token_type_id = tf.fill(dims=input_shape, value=0)
        hidden_states = self.embedding_decoder(
            input_ids=decoder_inputs, token_type_ids=decoder_token_type_id)
        decoder_output = self.decoder(
            hidden_states=hidden_states,
            attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=None,
            training=training,
        )

        logits = self.logit(decoder_output)
        start_logits, end_logits = tf.split(
            value=logits, num_or_size_splits=2, axis=-1)
        start_logits = self.start(tf.squeeze(input=start_logits, axis=-1))
        end_logits = self.end(tf.squeeze(input=end_logits, axis=-1))
        return start_logits, end_logits


# ==============================================================
app = Flask(__name__)
api = Api(app)

tokenize = TokenizeForBert()

maxlen_c = 702
maxlen_q = 100
config = Config()
config.input_vocab_size = tokenize.vocab_size
config.target_vocab_size = tokenize.vocab_size
model = TFRobertaModel(config, maxlen_c)

checkpoint_path = "checkpoint"

ckpt = tf.train.Checkpoint(model=model)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=2)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)  # .expect_partial()
    print('Latest checkpoint restored!!')

corenlp = VnCoreNLP(address="http://127.0.0.1", port=9000)


class Chat(Resource):
    def __init__(self) -> None:
        super().__init__()

    def post(self):
        question = request.form['question']
        question = remove_emoji(question)
        question = convert_unicode(question)
        question = vncore_tokenize(question, corenlp).lower()
        # =======================
        context = request.form['context']
        context = remove_emoji(context)
        context = convert_unicode(context)
        context = vncore_tokenize(context, corenlp).lower()

        q_token = tokenize([question], max_length=maxlen_q,
                           truncation='longest_first', padding='max_length')
        c_token = tokenize([context], max_length=800,
                           truncation='longest_first', padding='max_length')
        encode_input_ids = tf.convert_to_tensor(
            q_token['input_ids'], dtype=tf.int32)
        encode_attention_mask = tf.convert_to_tensor(
            q_token['attention_mask'], dtype=tf.int32)
        decode_input_ids = tf.convert_to_tensor(
            c_token['input_ids'], dtype=tf.int32)
        decode_attention_mask = tf.convert_to_tensor(
            c_token['attention_mask'], dtype=tf.int32)

        start, end = model(
            encoder_inputs=encode_input_ids,
            encoder_attention_mask=encode_attention_mask,
            encoder_token_type_id=None,
            head_mask=None,
            decoder_inputs=decode_input_ids,
            decoder_attention_mask=decode_attention_mask,
            decoder_token_type_id=None,
        )

        start = np.argmax(start.numpy()[0])
        end = np.argmax(end.numpy()[0])

        result = context.split()[start:end+1]

        return {
            'question': question,
            'result': ' '.join(result).replace('_', ' ')
        }


api.add_resource(Chat, '/')

if __name__ == '__main__':
    app.run(debug=True, port=8080)
