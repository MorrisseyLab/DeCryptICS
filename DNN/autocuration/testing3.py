q = np.array(range(32))
q = np.reshape(q, (2,1,1,-1))
Q = K.constant(q)

a = np.array(range(3*3*18))
a = np.reshape(a, (2,3,3,-1))
A = K.constant(a)
inputs = [Q,A]

#AttentionROI(self, Fout, dk, dff, Nh, dr, relative=False)
attroi = AttentionROI(Q.shape[-1], 2, 12, 2, 0.3)
attconv = AttentionAugmentation2D(Q.shape[-1], 3, 2, 6, 2, relative=True)

    def call(inputs):
        shape = K.int_shape(inputs[1])        
        if None in shape:
            shape = [-1 if type(v)==type(None) else v for v in shape]
        Batch, H, W, _ = shape
        tileby = tf.constant([1, H, W, 1], tf.int32)
#        q_exp = tf.tile(inputs[0], tileby)
#        flat_q_old, flat_k_old, flat_v_old = attroi.compute_flat_qkv([q_exp, inputs[1]])
        flat_q, flat_k, flat_v = compute_flat_qkv([inputs[0], inputs[1]])
        dkh = attroi.depth_k // attroi.num_heads
        # might be able to calculate this differently as query will be "1x1xdepth"
        # so same all over the spatial dimensions? (reduce size of problem)
        
#        logits = tf.matmul(flat_q_old, flat_k_old, transpose_b=True)
        logits_red = tf.matmul(flat_q, flat_k, transpose_b=True)

#        weights = K.softmax(logits, axis=-1)
        weights_red = K.softmax(logits_red, axis=-1)
#        attn_out = tf.matmul(weights, flat_v)
        attn_out_red = tf.matmul(weights_red, flat_v)
        
#        attn_out = K.reshape(attn_out, [Batch, attroi.num_heads, H, W, attroi.depth_v // attroi.num_heads])
        attn_out_red = K.reshape(attn_out_red, [Batch, attroi.num_heads, 1, 1, attroi.depth_v // attroi.num_heads])

#        attn_out = attroi.combine_heads_2d(attn_out)
        attn_out_red = attroi.combine_heads_2d(attn_out_red)
        attn_out_red = tf.tile(attn_out_red, tileby)
        A = attroi.attn_out_conv(attn_out_red)
        A = K.sum(A, axis=2, keepdims=True)
        A = K.sum(A, axis=1, keepdims=True)
        
        # combine with query
        A = attroi.dropout(A)
        Qp = attroi.add([inputs[0], A])
        Qp = attroi.laynorm(Qp)
        # FFN
        Qp_ff = attroi.c_ffn_1(Qp)
        Qp_ff = attroi.activ(Qp_ff)
        Qp_ff = attroi.c_ffn_2(Qp_ff)
        Qp_ff = attroi.dropout(Qp_ff)
        Qpp = attroi.add([Qp, Qp_ff])
        Qpp = attroi.laynorm(Qpp)        
        return Qpp

    def combine_heads_2d(x):
        # [batch, num_heads, height, width, depth_v // num_heads]
        transposed = K.permute_dimensions(x, [0, 2, 3, 1, 4])
        # [batch, height, width, num_heads, depth_v // num_heads]
        shape = K.int_shape(transposed)
        if None in shape:
            shape = [-1 if type(v)==type(None) else v for v in shape]
        batch, h, w, a, b = shape 
        ret_shape = [batch, h ,w, a*b]
        # [batch, height, width, depth_v]
        return K.reshape(transposed, ret_shape)

    def split_heads_2d(q,Nh):
        batch, height, width, channels = K.int_shape(q)
        ret_shape = [-1, height, width, Nh, channels//Nh]
        split = K.reshape(q,ret_shape)
        transpose_axes = (0, 3, 1, 2, 4)
        split = K.permute_dimensions(split, transpose_axes)
        return split

    def compute_flat_qkv(inputs):
        kv = attroi.kv(inputs[1])
        q = attroi.q_init(inputs[0])
        B, H, W, _ = K.int_shape(inputs[1])
        k, v = tf.split(kv, [attroi.depth_k, attroi.depth_v], axis=3)

        dkh = attroi.depth_k // attroi.num_heads
        dvh = attroi.depth_v // attroi.num_heads
        q *= dkh ** -0.5

        attroi.q = attroi.split_heads_2d(q, attroi.num_heads)
        attroi.k = attroi.split_heads_2d(k, attroi.num_heads)
        attroi.v = attroi.split_heads_2d(v, attroi.num_heads)

        flat_q = K.reshape(attroi.q, [-1, attroi.num_heads, 1    , dkh])
        flat_k = K.reshape(attroi.k, [-1, attroi.num_heads, H * W, dkh])
        flat_v = K.reshape(attroi.v, [-1, attroi.num_heads, H * W, dvh])

        return flat_q, flat_k, flat_v
    
    def compute_output_shape(input_shape):
        output_shape = list(input_shape[0])
        output_shape[-1] = attroi.output_filters
        return tuple(output_shape)
