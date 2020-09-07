import keras
import tensorflow as tf 
import numpy as np 

def scaled_dot_product_attention(q, k, v):
  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., )
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.sqrt(dk)
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., )
  output = tf.matmul(attention_weights, v)  # (..., )
  return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    assert d_model % self.num_heads == 0    
    self.depth = d_model // self.num_heads
    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)
    self.dense = tf.keras.layers.Dense(d_model)
  
  def split_heads(self, x):
    x = tf.reshape(x, (self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 1])
  
  def call(self, v, k, q):
    q = self.wq(q)  # (d_model)
    k = self.wk(k)  # (d_model)
    v = self.wv(v)  # (d_model)    
    q = self.split_heads(q)  # (num_heads, depth)
    k = self.split_heads(k)  # (num_heads, depth)
    v = self.split_heads(v)  # (num_heads, depth)
    # scaled_attention.shape == ( num_heads, depth)
    # attention_weights.shape == ( num_heads )
    scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v)
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 1])  #  (num_heads, depth)
    concat_attention = tf.reshape(scaled_attention, (-1, self.d_model))  # (d_model)
    output = self.dense(concat_attention)  # (d_model)       
    return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (dff)
      tf.keras.layers.Dense(d_model)  # (d_model)
  ])


class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()
    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    
  def call(self, x, training):
    attn_output, _ = self.mha(x, x, x)  # ( d_model)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = tf.contrib.layers.layer_norm(x + attn_output)
    ffn_output = self.ffn(out1)  # ( d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = tf.contrib.layers.layer_norm(out1 + ffn_output)
    return out2


class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()
    self.mha1 = MultiHeadAttention(d_model, num_heads)
    self.mha2 = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)
    
  def call(self, x, enc_output, training):
    # enc_output.shape == (batch_size, d_model)
    attn1, attn_weights_block1 = self.mha1(x, x, x)  # ( d_model)
    attn1 = self.dropout1(attn1, training=training)
    out1 = tf.contrib.layers.layer_norm(x + attn1)
    attn2, attn_weights_block2 = self.mha2(
        enc_output, enc_output, out1)  # ( d_model)
    attn2 = self.dropout2(attn2, training=training)
    out2 = tf.contrib.layers.layer_norm(attn2 + out1)
    ffn_output = self.ffn(out2)  # (batch_size, d_model)
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = tf.contrib.layers.layer_norm(ffn_output + out2)
    return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
    super(Encoder, self).__init__()
    self.d_model = d_model
    self.num_layers = num_layers
    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)
  
  def call(self, x, training):
    # adding embedding and position encoding.
    x *= tf.sqrt(tf.cast(self.d_model, tf.float32))
    x = self.dropout(x, training=training)
    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training)
    return x  # (d_model)


class Decoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
    super(Decoder, self).__init__()
    self.d_model = d_model
    self.num_layers = num_layers
    self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)
    
  def call(self, x, enc_output, training):
    seq_len = tf.shape(x)[0]
    attention_weights = {}
    x *= tf.sqrt(tf.cast(self.d_model, tf.float32))
    x = self.dropout(x, training=training)
    for i in range(self.num_layers):
      x, block1, block2 = self.dec_layers[i](x, enc_output, training)
      attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
      attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
    # x.shape == ( d_model)
    return x, attention_weights


class Transformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
    super(Transformer, self).__init__()
    self.encoder = Encoder(num_layers, d_model, num_heads, dff, rate)
    self.decoder = Decoder(num_layers, d_model, num_heads, dff, rate)
    self.final_layer = tf.keras.layers.Dense(d_model)
    
  def call(self, inp, tar, training):
    enc_output = self.encoder(inp, training)  # (batch_size, d_model)
    # dec_output.shape == (batch_size, d_model)
    dec_output, attention_weights = self.decoder(tar, enc_output, training)
    final_output = self.final_layer(dec_output)  # (batch_size, d_model)
    return final_output, attention_weights
