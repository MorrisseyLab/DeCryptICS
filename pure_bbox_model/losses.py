from typing import Union,Dict,Tuple
from itertools import product
import tensorflow as tf
import numpy as np
#import bbox
from bbox import xcycwh_to_xy_min_xy_max, merge, jaccard
from scipy.optimize import linear_sum_assignment

mtp = tf.keras.metrics.TruePositives()
mfn = tf.keras.metrics.FalseNegatives()
mfp = tf.keras.metrics.FalsePositives()
mtn = tf.keras.metrics.TrueNegatives()
eps = tf.constant(1e-8, dtype=tf.float32)

#loss_weights = [1, 2, 5] # label_cost, giou_loss, l1_loss
#class_weights = [1., 1.5, 2., 1.25, 1.] # crypt, clone, partial, fufi, no object

## test
#from scipy.special import logit
#im, bb_ps = train_datagen[0]
#bboxes = bb_ps[0]
#labels = bb_ps[1]
#p_bboxes = np.empty((bboxes.shape[0],bboxes.shape[1]-1,bboxes.shape[2]), dtype=bboxes.dtype)
#p_labels = np.empty((labels.shape[0],labels.shape[1]-1,labels.shape[2]), dtype=labels.dtype)
#shuff_inds = np.empty((p_bboxes.shape[0],p_bboxes.shape[1]), dtype=np.int32)
#for b in range(bboxes.shape[0]):
#   indx_shuffle = np.random.choice(p_bboxes.shape[1], p_bboxes.shape[1], replace=False)        
#   p_bboxes[b,:,:] = bboxes[b,1:,:][indx_shuffle,:]
#   p_labels[b,:,:] = labels[b,1:,:][indx_shuffle,:]
#   shuff_inds[b,:] = indx_shuffle
#p_bboxes = p_bboxes + np.random.normal(0,0.001,size=p_bboxes.shape).astype(bboxes.dtype)
#p_labels = p_labels + np.random.normal(0,0.2,size=p_labels.shape).astype(labels.dtype)
#p_labels[p_labels<0] = 0.00001
#p_labels[p_labels>=1] = 0.99
#p_labels = logit(p_labels).astype(labels.dtype)
#p_bboxes[p_bboxes<0] = 0
#p_bboxes[p_bboxes>1] = 1
#y_true = [tf.convert_to_tensor(bboxes), tf.convert_to_tensor(labels)]
#y_pred = [tf.convert_to_tensor(p_bboxes), tf.convert_to_tensor(p_labels)]


def np_tf_linear_sum_assignment(matrix):
    indices = linear_sum_assignment(matrix)
    target_indices = indices[0]
    pred_indices = indices[1]

    target_selector = np.zeros(matrix.shape[0])
    target_selector[target_indices] = 1
    target_selector = target_selector.astype(np.bool)

    pred_selector = np.zeros(matrix.shape[1])
    pred_selector[pred_indices] = 1
    pred_selector = pred_selector.astype(np.bool)

    return [target_indices, pred_indices, target_selector, pred_selector]

def hungarian_matching(t_bbox, t_class, p_bbox, p_class, lwts, slice_preds=True) -> tuple:
    if slice_preds:
        n_objs = tf.cast(t_bbox[0][0], tf.int32) # this takes the num_bboxes from the padded header row, containing (say) [400, 0, 0, 0], hence n_objs would be 400
        t_bbox = tf.slice(t_bbox, [1, 0], [n_objs, 4]) # slice begins at first row to get rid of the header row
        t_class = tf.slice(t_class, [1, 0], [n_objs, -1])
#        t_class = tf.squeeze(t_class, axis=-1)

    # Convert from [xc, yc, w, h] to [xmin, ymin, xmax, ymax]
    p_bbox_xy = xcycwh_to_xy_min_xy_max(p_bbox)
    t_bbox_xy = xcycwh_to_xy_min_xy_max(t_bbox)
#    t_bbox_xy = tf.cast(bbox.xcycwh_to_xy_min_xy_max(t_bbox), tf.float64)

    # Classification cost for the Hungarian algorithom 
    # On each prediction. We select the prob of the expected class
    sigmoid = tf.math.sigmoid(p_class) # this is because detr has exclusive classes, we don't!
    _p_class, _t_class = merge(sigmoid, t_class)
    cost_class = tf.norm(_p_class - _t_class, ord=1, axis=-1)
    ## we could introduce different class weightings by writing a custom
    ## norm fucntion above, multipling classes separately before summing

    # L1 cost for the hungarian algorithm
    _p_bbox, _t_bbox = merge(p_bbox, t_bbox)
    cost_bbox = tf.norm(_p_bbox - _t_bbox, ord=1, axis=-1)

    # Generalized IOU
    iou, union = jaccard(p_bbox_xy, t_bbox_xy, return_union=True)
    _p_bbox_xy, _t_bbox_xy = merge(p_bbox_xy, t_bbox_xy)
    top_left = tf.math.minimum(_p_bbox_xy[:,:,:2], _t_bbox_xy[:,:,:2])
    bottom_right =  tf.math.maximum(_p_bbox_xy[:,:,2:], _t_bbox_xy[:,:,2:])
    size = tf.nn.relu(bottom_right - top_left)
    area = size[:,:,0] * size[:,:,1]
    cost_giou = -(iou - (area - union) / area)

    # Final hungarian cost matrix
#    loss_weights = [1, 2, 5] # label_cost, giou_loss, l1_loss
    cost_matrix = lwts[2] * cost_bbox + lwts[0] * cost_class + lwts[1] * cost_giou

    selectors = tf.numpy_function(np_tf_linear_sum_assignment, [cost_matrix], [tf.int64, tf.int64, tf.bool, tf.bool] )
    target_indices = selectors[0]
    pred_indices = selectors[1]
    target_selector = selectors[2]
    pred_selector = selectors[3]
    return pred_indices, target_indices, pred_selector, target_selector, t_bbox, t_class

def calc_loss_alone(y_true, y_pred, loss_weights, class_weights):
   def curried_loss(x):
      losses = get_batch_losses([x[0], x[1]], [x[2], x[3]], loss_weights, class_weights)
      total_loss = get_total_loss(losses, loss_weights)
      return total_loss
   all_losses = tf.map_fn(curried_loss, [y_true[0], y_true[1], y_pred[0], y_pred[1]], dtype=tf.float32)   
   return tf.reduce_mean(all_losses, axis=-1)

def batched_loss_closure(y_true, y_pred, loss_weights, class_weights):
   def curried_loss(x):
      losses = get_batch_losses([x[0], x[1]], [x[2], x[3]], loss_weights, class_weights)
      total_loss = get_total_loss(losses, loss_weights)
      losses.update({'total_loss':total_loss})
      metrics = tf.stack(list(losses.values()), axis=0)
      return metrics
   all_losses = tf.map_fn(curried_loss, [y_true[0], y_true[1], y_pred[0], y_pred[1]], dtype=tf.float32)
   return all_losses

def get_losses(y_true, y_pred, loss_weights, class_weights):
   losses = get_my_losses(y_true, y_pred, loss_weights, class_weights)
   total_loss = get_total_loss(losses, loss_weights)
   return total_loss, losses

def get_total_loss(losses, loss_weights):
    train_names = ["label_loss", "giou_loss", "l1_loss"]
    total_loss = 0
    for key in losses:
        selector = [l for l, loss_name in enumerate(train_names) if loss_name in key]
        if len(selector) == 1:
            total_loss += losses[key]*loss_weights[selector[0]]
    return total_loss

def loss_boxes(p_bbox, p_class, t_bbox, t_class, t_indices, p_indices, t_selector, p_selector):
    p_bbox = tf.gather(p_bbox, p_indices)
    t_bbox = tf.gather(t_bbox, t_indices)

    p_bbox_xy = xcycwh_to_xy_min_xy_max(p_bbox)
    t_bbox_xy = xcycwh_to_xy_min_xy_max(t_bbox)

    l1_loss = tf.abs(p_bbox - t_bbox)
    l1_loss = tf.reduce_sum(l1_loss) / (tf.cast(tf.shape(p_bbox)[0], l1_loss.dtype) + eps)

    iou, union = jaccard(p_bbox_xy, t_bbox_xy, return_union=True)

    _p_bbox_xy, _t_bbox_xy = merge(p_bbox_xy, t_bbox_xy)
    top_left = tf.math.minimum(_p_bbox_xy[:,:,:2], _t_bbox_xy[:,:,:2])
    bottom_right =  tf.math.maximum(_p_bbox_xy[:,:,2:], _t_bbox_xy[:,:,2:])
    size = tf.nn.relu(bottom_right - top_left)
    area = size[:,:,0] * size[:,:,1]
    giou = (iou - (area - union) / (area + eps) )
    loss_giou = 1 - tf.linalg.diag_part(giou)

    loss_giou = tf.reduce_sum(loss_giou) / (tf.cast(tf.shape(p_bbox)[0], loss_giou.dtype) + eps)

    return loss_giou, l1_loss

def loss_labels(p_bbox, p_class, t_bbox, t_class, t_indices, p_indices, t_selector, p_selector, clswts):

    neg_indices = tf.squeeze(tf.where(p_selector == False), axis=-1)
    neg_p_class = tf.gather(p_class, neg_indices)
    neg_t_class = tf.concat([tf.zeros((tf.shape(neg_p_class)[0], tf.shape(neg_p_class)[1]-1), p_class.dtype), tf.ones((tf.shape(neg_p_class)[0], 1), p_class.dtype)], axis=1)
    
    # weights between null boxes and true boxes
    neg_weights = tf.concat([tf.ones((tf.shape(neg_indices)[0], 1), p_class.dtype)*clswts[0],
                             tf.ones((tf.shape(neg_indices)[0], 1), p_class.dtype)*clswts[1],
                             tf.ones((tf.shape(neg_indices)[0], 1), p_class.dtype)*clswts[2],
                             tf.ones((tf.shape(neg_indices)[0], 1), p_class.dtype)*clswts[3],
                             tf.ones((tf.shape(neg_indices)[0], 1), p_class.dtype)*clswts[4]], axis=1) * 0.1
    pos_weights = tf.concat([tf.ones((tf.shape(t_indices)[0], 1), p_class.dtype)*clswts[0],
                             tf.ones((tf.shape(t_indices)[0], 1), p_class.dtype)*clswts[1],
                             tf.ones((tf.shape(t_indices)[0], 1), p_class.dtype)*clswts[2],
                             tf.ones((tf.shape(t_indices)[0], 1), p_class.dtype)*clswts[3],
                             tf.ones((tf.shape(t_indices)[0], 1), p_class.dtype)*clswts[4]], axis=1) * 1.0
    
    weights = tf.concat([neg_weights, pos_weights], axis=0)
    
    # this subsets to the indices matched
    pos_p_class = tf.gather(p_class, p_indices)
    pos_t_class = tf.gather(t_class, t_indices)
    
    # loss
    targets = tf.concat([neg_t_class, pos_t_class], axis=0)
    preds = tf.concat([neg_p_class, pos_p_class], axis=0)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=preds)
    loss = tf.reduce_sum(loss * weights) / tf.reduce_sum(weights)

    ## metrics
    _preds = tf.math.sigmoid(preds)
#    mtp = tf.keras.metrics.TruePositives()
#    mfn = tf.keras.metrics.FalseNegatives()
#    mfp = tf.keras.metrics.FalsePositives()
#    mtn = tf.keras.metrics.TrueNegatives()
    cl_tp = []
    cl_fn = []
    cl_fp = []
    cl_tn = []
    cl_precision = []
    cl_recall = []
    cl_accuracy = []
    for cl in range(len(clswts)):
       mtp.update_state(targets[:,cl], _preds[:,cl])
       cl_tp.append(mtp.result())
       mtp.reset_states()

       mfn.update_state(targets[:,cl], _preds[:,cl])
       cl_fn.append(mfn.result())
       mfn.reset_states()
       
       mfp.update_state(targets[:,cl], _preds[:,cl])
       cl_fp.append(mfp.result())
       mfp.reset_states()
       
       mtn.update_state(targets[:,cl], _preds[:,cl])
       cl_tn.append(mtn.result())
       mtn.reset_states()
       
       cl_precision.append(precision(cl_tp[-1],cl_fp[-1]))
       cl_recall.append(recall(cl_tp[-1],cl_fn[-1]))
       cl_accuracy.append(accuracy(cl_tp[-1],cl_fn[-1],cl_fp[-1],cl_tn[-1]))

    return loss, cl_tp, cl_fn, cl_fp, cl_tn, cl_precision, cl_recall, cl_accuracy

def precision(tp,fp):
    return tp/(tp + fp + eps)

def recall(tp,fn):
    return tp/(tp + fn + eps)

def accuracy(tp,fn,fp,tn):
    return (tp + tn)/(tp + tn + fp + fn)

def get_my_losses(y_true, y_pred, loss_weights, class_weights):
    target_bbox = y_true[0]
    target_label = y_true[1]
    predicted_bbox = y_pred[0]
    predicted_label = y_pred[1]

    all_target_bbox = []
    all_target_class = []
    all_predicted_bbox = []
    all_predicted_class = []
    all_target_indices = []
    all_predcted_indices = []
    all_target_selector = []
    all_predcted_selector = []

    t_offset = 0
    p_offset = 0
    for b in tf.range(tf.shape(predicted_bbox)[0]): # cycle through batch dim
        p_bbox, p_class, t_bbox, t_class = predicted_bbox[b], predicted_label[b], target_bbox[b], target_label[b]
        t_indices, p_indices, t_selector, p_selector, t_bbox, t_class = hungarian_matching(t_bbox, t_class, p_bbox, p_class, loss_weights, slice_preds=True)

        t_indices = t_indices + tf.cast(t_offset, tf.int64)
        p_indices = p_indices + tf.cast(p_offset, tf.int64)

        all_target_bbox.append(t_bbox)
        all_target_class.append(t_class)
        all_predicted_bbox.append(p_bbox)
        all_predicted_class.append(p_class)
        all_target_indices.append(t_indices)
        all_predcted_indices.append(p_indices)
        all_target_selector.append(t_selector)
        all_predcted_selector.append(p_selector)

        t_offset += tf.shape(t_bbox)[0]
        p_offset += tf.shape(p_bbox)[0]

    all_target_bbox = tf.concat(all_target_bbox, axis=0)
    all_target_class = tf.concat(all_target_class, axis=0)
    all_predicted_bbox = tf.concat(all_predicted_bbox, axis=0)
    all_predicted_class = tf.concat(all_predicted_class, axis=0)
    all_target_indices = tf.concat(all_target_indices, axis=0)
    all_predcted_indices = tf.concat(all_predcted_indices, axis=0)
    all_target_selector = tf.concat(all_target_selector, axis=0)
    all_predcted_selector = tf.concat(all_predcted_selector, axis=0)

    label_cost, true_pos, false_neg, false_pos, true_neg, prec, rec, acc = loss_labels(
        all_predicted_bbox,
        all_predicted_class,
        all_target_bbox,
        all_target_class,
        all_target_indices,
        all_predcted_indices,
        all_target_selector,
        all_predcted_selector,
        class_weights
    )

    giou_loss, l1_loss = loss_boxes(
        all_predicted_bbox,
        all_predicted_class,
        all_target_bbox,
        all_target_class,
        all_target_indices,
        all_predcted_indices,
        all_target_selector,
        all_predcted_selector
    )

    label_cost = label_cost
    giou_loss = giou_loss
    l1_loss = l1_loss
    outdict = {
        "label_loss": label_cost,
        "giou_loss": giou_loss,
        "l1_loss": l1_loss
        }
    class_names = ['crypt', 'clone', 'partial', 'fufi', 'no_object']
    for i, cl in enumerate(class_names):
       cldict = {        
           "true_pos_{}".format(cl): true_pos[i],
           "false_neg_{}".format(cl): false_neg[i],
           "false_pos_{}".format(cl): false_pos[i],
           "true_neg_{}".format(cl): true_neg[i],
           "precision_{}".format(cl): prec[i],
           "recall_{}".format(cl): rec[i],
           "accuracy_{}".format(cl): acc[i]
       }
       outdict.update(cldict)
    return outdict

def get_batch_losses(y_true, y_pred, loss_weights, class_weights):
    t_bbox = y_true[0]
    t_class = y_true[1]
    p_bbox = y_pred[0]
    p_class = y_pred[1]

    t_indices, p_indices, t_selector, p_selector, t_bbox, t_class = hungarian_matching(t_bbox, t_class, p_bbox, p_class, loss_weights, slice_preds=True)

    label_cost, true_pos, false_neg, false_pos, true_neg, prec, rec, acc = loss_labels(
        p_bbox,
        p_class,
        t_bbox,
        t_class,
        t_indices,
        p_indices,
        t_selector,
        p_selector,
        class_weights
    )

    giou_loss, l1_loss = loss_boxes(
        p_bbox,
        p_class,
        t_bbox,
        t_class,
        t_indices,
        p_indices,
        t_selector,
        p_selector
    )

    label_cost = label_cost
    giou_loss = giou_loss
    l1_loss = l1_loss
    outdict = {
        "label_loss": label_cost,
        "giou_loss": giou_loss,
        "l1_loss": l1_loss
        }
    class_names = ['crypt', 'clone', 'partial', 'fufi', 'no_object']
    for i, cl in enumerate(class_names):
       cldict = {        
           "true_pos_{}".format(cl): true_pos[i],
           "false_neg_{}".format(cl): false_neg[i],
           "false_pos_{}".format(cl): false_pos[i],
           "true_neg_{}".format(cl): true_neg[i],
           "precision_{}".format(cl): prec[i],
           "recall_{}".format(cl): rec[i],
           "accuracy_{}".format(cl): acc[i]
       }
       outdict.update(cldict)
    return outdict

class DummyMetric1(tf.keras.metrics.Metric):
    def __init__(self, name='', **kwargs):
      super(DummyMetric1, self).__init__(name=name, **kwargs)
      self.total = self.add_weight(name='total', initializer='zeros')
    def update_state(self, new_state):
      self.total.assign_add(tf.reduce_sum(new_state))
    def result(self):
      return self.total
      
class DummyMetric2(tf.keras.metrics.Metric):
    def __init__(self, name='', **kwargs):
      super(DummyMetric2, self).__init__(name=name, **kwargs)
      self.total = self.add_weight(name='total', initializer='zeros')
      self.count = self.add_weight(name='count', initializer='zeros')
    def update_state(self, new_state):
      self.total.assign_add(tf.reduce_sum(new_state))
      self.count.assign_add(tf.constant(1, dtype=tf.float32))
    def result(self):
      return tf.divide(self.total, self.count)


