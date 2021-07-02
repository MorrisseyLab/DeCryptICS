import os
import tensorflow as tf
import pandas as pd
from training.losses import get_total_loss, batched_loss_closure, DummyMetric1, DummyMetric2

def setup_optimizer(model, lr, clipnorm=True):
   return {
     "optimizer": tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=clipnorm),
     "variables": model.trainable_variables
   }

def gather_gradient(model, total_loss, tape):
    trainable_variables = model.trainable_variables
    gradients = tape.gradient(total_loss, trainable_variables)
    return gradients

def apply_gradient(grads, optimizer):
   optimizer["gradients"] = grads
   optimizer["optimizer"].apply_gradients(zip(optimizer["gradients"], optimizer["variables"]))

def create_loss_metric_trackers(loss_names, class_names):
   loss_trackers = {}
   for i, cl in enumerate(loss_names):
       cldict = {        
           "{}".format(cl): DummyMetric2(name="{}".format(cl)),
       }
       loss_trackers.update(cldict)
   for i, cl in enumerate(class_names):
       cldict = {        
           "true_pos_{}".format(cl): DummyMetric1(name="true_pos_{}".format(cl)),
           "false_neg_{}".format(cl): DummyMetric1(name="false_neg_{}".format(cl)),
           "false_pos_{}".format(cl): DummyMetric1(name="false_pos_{}".format(cl)),
           "true_neg_{}".format(cl): DummyMetric1(name="true_neg_{}".format(cl)),
           "precision_{}".format(cl): DummyMetric2(name="precision_{}".format(cl)),
           "recall_{}".format(cl): DummyMetric2(name="recall_{}".format(cl)),
           "accuracy_{}".format(cl): DummyMetric2(name="accuracy_{}".format(cl))
       }
       loss_trackers.update(cldict)
   return loss_trackers

def update_trackers(losses, loss_trackers, all_names):
   for i, cl in enumerate(all_names):
      if type(loss_trackers[cl])==DummyMetric1:
         loss_trackers[cl].update_state(tf.reduce_sum(losses[:,i]))
      if type(loss_trackers[cl])==DummyMetric2:
         loss_trackers[cl].update_state(tf.reduce_mean(losses[:,i]))
   return loss_trackers
   
def log_trackers(loss_trackers, all_names, loss_names, loss_weights, epoch, path, suffix=''):
   outdict = {}
   for i, cl in enumerate(all_names):
      outdict.update({'{}'.format(cl) : float(loss_trackers[cl].result())})
   outdict.update({'total_loss': aggregate_losses(loss_trackers, loss_names, loss_weights), 
                   'epoch': epoch})
   fout = path + '_' + suffix + '.csv'
   df = pd.DataFrame(outdict, index = [epoch])
   if not os.path.isfile(fout):
      df.to_csv(fout, header='column_names', index=False)
   else:
      df.to_csv(fout, mode='a', header=False, index=False)

def reset_trackers(loss_trackers, all_names):
    for i, cl in enumerate(all_names):
       loss_trackers[cl].reset_states()
    return loss_trackers

def aggregate_losses(loss_trackers, loss_names, loss_weights):
    total_loss = 0
    for key in loss_trackers:
        selector = [l for l, loss_name in enumerate(loss_names) if loss_name in key]
        if len(selector) == 1:
            total_loss += loss_trackers[key].result()*loss_weights[selector[0]]
    return float(total_loss)

