def call(inputs):
  out = attconv.conv_out(inputs)
  shape = K.int_shape(inputs)
  if None in shape:
      shape = [-1 if type(v)==type(None) else v for v in shape]
  Batch,H,W,_ = shape

  flat_q, flat_k, flat_v = attconv.compute_flat_qkv(inputs)
  dkh = attconv.depth_k // attconv.num_heads
  # might be able to calculate this differently as query will be "1x1xdepth"
  # so same all over the spatial dimensions? (reduce size of problem)
  logits = tf.matmul(flat_q, flat_k, transpose_b=True)
  if attconv.relative:
      h_rel_logits, w_rel_logits = attconv.relative_logits(attconv.q, H, W, attconv.num_heads,dkh)
      logits += h_rel_logits
      logits += w_rel_logits
  
  weights = K.softmax(logits, axis=-1)
  attn_out = tf.matmul(weights, flat_v)
  attn_out = K.reshape(attn_out, [Batch, attconv.num_heads, H, W, attconv.depth_v // attconv.num_heads])

  attn_out = attconv.combine_heads_2d(attn_out)
  attn_out = attconv.attn_out_conv(attn_out)
  output =  concatenate([out,attn_out],axis=3)
  return output

def combine_heads_2d(x):
  # [batch, num_heads, height, width, depth_v // num_heads]
  transposed = K.permute_dimensions(x, [0, 2, 3, 1, 4])
  # [batch, height, width, num_heads, depth_v // num_heads]
  shape = K.int_shape(transposed)
  if None in shape:
      shape = [-1 if type(v)==type(None) else v for v in shape]
  batch, h , w, a , b = shape 
  ret_shape = [batch, h ,w, a*b]
  # [batch, height, width, depth_v]
  return K.reshape(transposed, ret_shape)

def rel_to_abs(x):
  shape = K.shape(x)
  shape = [shape[i] for i in range(3)]
  B, Nh, L = shape
  col_pad = K.zeros((B, Nh, L, 1), name="zero1")
  x = K.concatenate([x, col_pad], axis=3)
  flat_x = K.reshape(x, [B, Nh, L * 2 * L])
  flat_pad = K.zeros((B, Nh, L-1), name="zero2")
  flat_x_padded = K.concatenate([flat_x, flat_pad], axis=2)
  final_x = K.reshape(flat_x_padded, [B, Nh, L+1, 2*L-1])
  final_x = final_x[:, :, :L, L-1:]
  return final_x

def relative_logits_1d(q, rel_k, H, W, Nh, transpose_mask):
  rel_logits = tf.einsum('bhxyd,md->bhxym', q, rel_k)
  rel_logits = K.reshape(rel_logits, [-1, Nh*H, W, 2*W-1])
  rel_logits = attconv.rel_to_abs(rel_logits)
  rel_logits = K.reshape(rel_logits, [-1, Nh, H, W, W])
  rel_logits = K.expand_dims(rel_logits, axis=3)
  rel_logits = K.tile(rel_logits, [1, 1, 1, H, 1, 1])
  rel_logits = K.permute_dimensions(rel_logits, transpose_mask)
  rel_logits = K.reshape(rel_logits, [-1, Nh, H*W, H*W])
  return rel_logits

def relative_logits(q, H, W, Nh, dkh):
  key_rel_w  = K.random_normal(shape = (int(2 * W - 1), dkh))
  key_rel_h  = K.random_normal(shape = (int(2 * H - 1), dkh))

  rel_logits_w = attconv.relative_logits_1d(q, key_rel_w, H, W, Nh, [0, 1, 2, 4, 3, 5])

  rel_logits_h = attconv.relative_logits_1d(K.permute_dimensions(q, [0, 1, 3, 2, 4]), 
                                            key_rel_h, W, H, Nh, [0, 1, 4, 2, 5, 3])
  return rel_logits_h , rel_logits_w

def split_heads_2d(q,Nh):
  batch, height,width,channels = K.int_shape(q)
  ret_shape = [-1,height,width,Nh,channels//Nh]
  split = K.reshape(q,ret_shape)
  transpose_axes = (0, 3, 1, 2, 4)
  split = K.permute_dimensions(split, transpose_axes)
  return split

def compute_flat_qkv(inputs):
  qkv = attconv.qkv(inputs)
  B,H,W,_ = K.int_shape(inputs)
  q,k,v = tf.split(qkv,[attconv.depth_k,attconv.depth_k,attconv.depth_v],axis=3)

  dkh = attconv.depth_k // attconv.num_heads
  dvh = attconv.depth_v // attconv.num_heads
  q *= dkh ** -0.5

  attconv.q = attconv.split_heads_2d(q, attconv.num_heads)
  attconv.k = attconv.split_heads_2d(k, attconv.num_heads)
  attconv.v = attconv.split_heads_2d(v, attconv.num_heads)

  flat_q = K.reshape(attconv.q, [-1, attconv.num_heads, H * W, dkh])
  flat_k = K.reshape(attconv.k, [-1, attconv.num_heads, H * W, dkh])
  flat_v = K.reshape(attconv.v, [-1, attconv.num_heads, H * W, dvh])

  return flat_q, flat_k, flat_v
