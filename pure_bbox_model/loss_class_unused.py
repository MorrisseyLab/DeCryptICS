from typing import Union,Dict,Tuple
from itertools import product
import tensorflow as tf
import numpy as np
import bbox
from scipy.optimize import linear_sum_assignment
from tensorflow.keras.losses import Loss
from training.losses import get_batch_losses, get_total_loss
loss_weights = [1, 2, 5] # label_cost, giou_loss, l1_loss
class_weights = [1., 1.5, 2., 1.25, 1.] # crypt, clone, partial, fufi, no object

## test
#from training.gen_v3 import DataGen_curt, CloneGen_curt, CloneFufiGen_curt
#from model_set_parameter_dicts import set_params
#params = set_params()
#dnnfolder = '/home/doran/Work/py_code/new_DeCryptICS/newfiles'
#logs_folder = dnnfolder + '/logs'
#epochs                      = 200
#params_gen_ed['umpp']       = 1.1
#params_gen_ed['num_bbox']   = 400 # 400
#params_gen_ed['batch_size'] = 8 # 16, was 7
#params_gen_ed['just_clone'] = False
#params_gen_ed['cpfr_frac']  = [1,1,1,1]
#nsteps        = 2
#nclone_factor = 2
#npartial_mult = 2
#already_curated = pd.read_csv('./training/manual_curation_files/curated_files_summary.txt', 
#                              names = ['file_name', 'slide_crtd'])
#already_curated = already_curated[already_curated['slide_crtd'] != 'cancel']
#bad_slides = pd.read_csv('./training/manual_curation_files/slidequality_scoring.csv')
#radiother = np.where(bad_slides['quality_label']==2)[0]
#staining = np.where(bad_slides['quality_label']==0)[0]
#dontuse = np.asarray(bad_slides['path'])[np.hstack([radiother, staining])]
#dontuse = pd.DataFrame({'file_name':list(dontuse)}).drop_duplicates(keep='first')
#inds = ~already_curated.file_name.isin(dontuse.file_name)
#already_curated = already_curated[inds]
#train_data      = already_curated.sample(150, random_state=22)
#train_datagen = DataGen_curt(params_gen_ed, train_data, nsteps, nclone_factor, npartial_mult)

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


class MultiLabelBoxLoss(Loss):
  """ Derived from Loss base class
  To be implemented by subclasses:
  * `call()`: Contains the logic for loss calculation using `y_true`, `y_pred`.
  Example subclass implementation:
  ```python
  class MeanSquaredError(Loss):
    def call(self, y_true, y_pred):
      y_pred = tf.convert_to_tensor_v2(y_pred)
      y_true = tf.cast(y_true, y_pred.dtype)
      return tf.reduce_mean(math_ops.square(y_pred - y_true), axis=-1)
  ```
  """

  def __init__(self, loss_weights=loss_weights, class_weights=class_weights):
    """Initializes `MultiLabelBoxLoss` class.
    Args:
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`.
      name: Optional name for the op.
    """
    super(MultiLabelBoxLoss, self).__init__()
    self.loss_weights  = loss_weights
    self.class_weights = class_weights

#  @abc.abstractmethod
#  @doc_controls.for_subclass_implementers
  def call(self, y_true, y_pred):
      """Invokes the `Loss` instance.
      Args:
      y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`, except
         sparse loss functions such as sparse categorical crossentropy where
         shape = `[batch_size, d0, .. dN-1]`
      y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`
      Returns:
      Loss values with the shape `[batch_size, d0, .. dN-1]`.
      """
      print(y_true.shape)
#      print(y_true[1].shape)
      print(y_pred.shape)
#      print(y_pred[1].shape)
      def curried_loss(x):
         losses = get_batch_losses([x[0], x[1]], [x[2], x[3]], self.class_weights)
         total_loss = get_total_loss(losses, self.loss_weights)
         return total_loss
      batch_loss = tf.map_fn(curried_loss, [y_true[0], y_true[1], y_pred[0], y_pred[1]], dtype=tf.float32)
      return batch_loss
      

