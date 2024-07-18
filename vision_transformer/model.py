# code copied from: https://github.com/emla2805/vision-transformer/blob/master/model.py

import tensorflow as tf
from keras import layers, Sequential, Model, ops


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f'embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}'
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = ops.cast(ops.shape(key)[-1], tf.float32)
        scaled_score = score / ops.sqrt(dim_key)
        weights = ops.nn.softmax(scaled_score, axis=-1)
        output = ops.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = ops.reshape(
            x, (batch_size, -1, self.num_heads, self.projection_dim)
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = ops.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = ops.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )
        output = self.combine_heads(concat_attention)
        return output


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.mlp = Sequential(
            [
                layers.Dense(mlp_dim, activation='gelu'),
                layers.Dropout(dropout),
                layers.Dense(embed_dim),
                layers.Dropout(dropout),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, inputs):
        inputs_norm = self.layernorm1(inputs)
        attention_output = self.attention(inputs_norm)
        attention_output = self.dropout1(attention_output)
        out1 = attention_output + inputs

        out1_norm = self.layernorm2(out1)
        mlp_output = self.mlp(out1_norm)
        mlp_output = self.dropout2(mlp_output)
        return mlp_output + out1


class VisionTransformer(Model):
    def __init__(
        self,
        image_size,
        patch_size,
        num_layers,
        num_classes,
        d_model,
        num_heads,
        mlp_dim,
        channels,
        dropout,
    ):
        super(VisionTransformer, self).__init__()
        num_patches = (image_size // patch_size) ** 2
        self.patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_emb = self.add_weight(
            name='pos_emb', shape=(1, num_patches + 1, d_model)
        )
        self.class_emb = self.add_weight(name='class_emb', shape=(1, 1, d_model))
        self.patch_proj = layers.Dense(d_model)
        self.enc_layers = [
            TransformerBlock(d_model, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ]
        self.mlp_head = Sequential(
            [
                layers.LayerNormalization(epsilon=1e-6),
                layers.Dense(mlp_dim, activation='gelu'),
                layers.Dropout(dropout),
                layers.Dense(num_classes),
            ]
        )

    def extract_patches(self, images):
        batch_size = ops.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )
        patches = ops.reshape(patches, [batch_size, -1, self.patch_dim])
        return patches

    def call(self, x):
        batch_size = ops.shape(x)[0]
        patches = self.extract_patches(x)
        x = self.patch_proj(patches)

        class_emb = ops.broadcast_to(
            self.class_emb, [batch_size, 1, self.d_model]
        )
        x = ops.concatenate([class_emb, x], axis=1)
        x = x + self.pos_emb

        for layer in self.enc_layers:
            x = layer(x)

        x = self.mlp_head(x[:, 0])
        return x
