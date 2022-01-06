'''Main function for UCI letter and spam datasets.
'''

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd

from data_loader import data_loader
# from gain import gain
from utils import rmse_loss, mae_loss
from progress_bar import InitBar
import tensorflow as tf
from tqdm import tqdm

from utils import normalization, renormalization, rounding
from utils import xavier_init
from utils import binary_sampler, uniform_sampler, sample_batch_index
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
gpu_options = tf.GPUOptions(allow_growth=True)



def gain(all_data_original, gain_parameters, Initial_num, thre_value, epsilon, value, s_miss):
  all_data_original = np.array(all_data_original)
  start = time.clock()


  all_data_original_m = 1 - np.isnan(all_data_original)

  all_norm_original_data, all_norm_original_parameters = normalization(all_data_original)
  all_norm_original_data_x = np.nan_to_num(all_norm_original_data, 0)

  Validation_num = Initial_num

  CLIP = [-0.01, 0.01]

  all_data_list = np.random.randint(len(all_data_original), size=int(len(all_data_original) * s_miss))
  all_data = all_data_original[all_data_list]

  num_list = np.random.randint(len(all_data), size=Initial_num + Validation_num)
  initial_list = num_list[:Initial_num]
  validation_list = num_list[Initial_num:]
  data_val = all_data[validation_list]
  data_m_val = 1 - np.isnan(data_val)

  data_x = all_data[initial_list]
  all_data_m = 1 - np.isnan(all_data)

  # Define mask matrix on initial data
  data_m = 1 - np.isnan(data_x)

  # System parameters
  batch_size = gain_parameters['batch_size']
  hint_rate = gain_parameters['hint_rate']
  alpha = gain_parameters['alpha']
  epoch = gain_parameters['epoch']
  guarantee = gain_parameters['guarantee']
  Sinkhorn_iter = 20

  # Other parameters
  no, dim = data_x.shape
  print(no, dim)

  # Hidden state dimensions
  h_dim = int(dim)

  epsilon_value = (np.exp(5 / epsilon) / np.power(epsilon, np.floor(dim / 2))) ** 2
  print('epsilon:', epsilon, 'epsilon_value:', epsilon_value)

  # Normalization
  norm_data, norm_parameters = normalization(data_x)
  norm_data_x = np.nan_to_num(norm_data, 0)

  val_norm_data, val_norm_parameters = normalization(data_val)
  val_norm_data_x = np.nan_to_num(val_norm_data, 0)

  all_norm_data, all_norm_parameters = normalization(all_data)
  all_norm_data_x = np.nan_to_num(all_norm_data, 0)

  ## GAIN architecture
  # Input placeholders
  # Data vector
  tf.compat.v1.disable_eager_execution()
  X = tf.compat.v1.placeholder(tf.float32, shape=[None, dim])
  # Mask vector
  M = tf.compat.v1.placeholder(tf.float32, shape=[None, dim])
  # Hint vector
  H = tf.compat.v1.placeholder(tf.float32, shape=[None, dim])
  # 'n' number
  # n_num = tf.compat.v1.placeholder(tf.float32, shape=[None, dim])
  n_num = tf.Variable(1)

  # Discriminator variables
  D_W1 = tf.Variable(xavier_init([dim * 2, h_dim]))  # Data + Hint as inputs
  D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

  D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
  D_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

  D_W3 = tf.Variable(xavier_init([h_dim, dim]))
  D_b3 = tf.Variable(tf.zeros(shape=[dim]))  # Multi-variate outputs

  theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

  # Generator variables
  # Data + Mask as inputs (Random noise is in missing components)
  G_W1 = tf.Variable(xavier_init([dim * 2, h_dim]))
  G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

  G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
  G_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

  G_W3 = tf.Variable(xavier_init([h_dim, dim]))
  G_b3 = tf.Variable(tf.zeros(shape=[dim]))

  theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

  def cost_matrix(x, y, m, p=2):
    "Returns the cost matrix C_{ij}=|x_i - y_j|^p"
    x_col = tf.expand_dims(x, 1)
    m_x_col = tf.expand_dims(m, 1)
    y_lin = tf.expand_dims(y, 0)
    m_y_lin = tf.expand_dims(m, 0)
    c = tf.reduce_sum((tf.abs(x_col * m_x_col - y_lin * m_y_lin)) ** p, axis=2)
    return c

  def sinkhorn_loss(x, y, m, epsilon, n, niter, p=2):
    # The Sinkhorn algorithm takes as input three variables :
    C = cost_matrix(x, y, m, p=p)  # Wasserstein cost function
    # both marginals are fixed with equal weights
    mu = tf.constant(1.0 / n, shape=[n])
    nu = tf.constant(1.0 / n, shape=[n])
    # Elementary operations
    def M(u, v):
      return (-C + tf.expand_dims(u, 1) + tf.expand_dims(v, 0)) / epsilon

    def lse(A):
      return tf.reduce_logsumexp(A, axis=1, keepdims=True)

    # Actual Sinkhorn loop
    u, v = 0. * mu, 0. * nu
    for i in range(niter):
      u = epsilon * (tf.math.log(mu) - tf.squeeze(lse(M(u, v)))) + u
      v = epsilon * (tf.math.log(nu) - tf.squeeze(lse(tf.transpose(M(u, v))))) + v

    u_final, v_final = u, v
    pi = tf.exp(M(u_final, v_final))
    cost = tf.reduce_sum(pi * C)
    return cost

  ## GAIN functions
  # Generator
  def generator(x, m):
    # Concatenate Mask and Data
    inputs = tf.concat(values=[x, m], axis=1)
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
    # MinMax normalized output
    G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)
    return G_prob

  def generator_nN(x, m, N_parameter):
    # Concatenate Mask and Data
    inputs = tf.concat(values=[x, m], axis=1)
    G_h1 = tf.nn.relu(tf.matmul(inputs, N_parameter[0]) + N_parameter[3])
    G_h2 = tf.nn.relu(tf.matmul(G_h1, N_parameter[1]) + N_parameter[4])
    # MinMax normalized output
    G_prob = tf.nn.sigmoid(tf.matmul(G_h2, N_parameter[2]) + N_parameter[5])
    return G_prob

  # Discriminator
  def discriminator(x, h):
    # Concatenate Data and Hint
    inputs = tf.concat(values=[x, h], axis=1)
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_logit = tf.matmul(D_h2, D_W3) + D_b3
    #D_prob = tf.nn.sigmoid(D_logit)
    return D_logit

  ## GAIN structure
  # Generator
  G_sample = generator(X, M)

  Sinkhorn_loss = sinkhorn_loss(X, G_sample, M, epsilon, batch_size, Sinkhorn_iter)

  # Combine with observed data
  Hat_X = X * M + G_sample * (1 - M)

  # Discriminator
  D_prob = discriminator(Hat_X, H)

  ## GAIN loss
  D_loss_temp = -tf.reduce_mean(M * (D_prob + 1e-8) + (1 - M) * (1. - D_prob + 1e-8))
  G_loss_temp = -tf.reduce_mean((1 - M) * (D_prob + 1e-8))
  MSE_loss = tf.reduce_mean((M * X - M * G_sample) ** 2) / tf.reduce_mean(M)
  D_loss = D_loss_temp
  G_loss = G_loss_temp + value * Sinkhorn_loss
  # G_loss = G_loss_temp + alpha * MSE_loss

  ####### Fast Hessians Calculation
  H_gradient_ = tf.gradients(G_loss, theta_G)
  # Hessians = tf.hessians(G_loss, theta_G)
  Len_matrix = len(theta_G)
  final = [tf.reshape(H_gradient_[i], [-1, 1]) for i in range(Len_matrix)]
  final_zhuan = [tf.transpose(tf.reshape(H_gradient_[i], [-1, 1])) for i in range(Len_matrix)]
  Fast_Hessians = [final[i] * final_zhuan[i] for i in range(Len_matrix)]
  # Hessians = [Fast_Hessians[num] + tf.eye(tf.shape(Fast_Hessians[num])[0]) * 10e-4 for num in range(Len_matrix)]

  Hessians = [Fast_Hessians[num] for num in range(Len_matrix)]
  H_hessians = []
  H_hessians.append(tf.reshape(Hessians[0], [2 * dim * dim, 2 * dim * dim]) + tf.eye(2 * dim * dim))
  H_hessians.append(tf.reshape(Hessians[1], [dim * dim, dim * dim]) + tf.eye(dim * dim))
  H_hessians.append(tf.reshape(Hessians[2], [dim * dim, dim * dim]) + tf.eye(dim * dim))
  H_hessians.append(tf.reshape(Hessians[3], [dim, dim]) + tf.eye(dim))
  H_hessians.append(tf.reshape(Hessians[4], [dim, dim]) + tf.eye(dim))
  H_hessians.append(tf.reshape(Hessians[5], [dim, dim]) + tf.eye(dim))

  H_invert = [tf.linalg.inv(item) for item in H_hessians]

  H_gradient_diag = [tf.linalg.tensor_diag_part(item) for item in H_invert]
  Mean_N = [tf.reshape(item, [-1]) for item in theta_G]
  # parameter settings
  Variance_N = [item * (1 / tf.cast(n_num, tf.float32) - 1/len(all_data)) for item in H_gradient_diag]
  Variance_n = [item * (1 / Initial_num - 1 / tf.cast(n_num, tf.float32)) for item in H_gradient_diag]

  n_parameter_noreshape = [tf.random.normal(tf.shape(Mean_N[i]), Mean_N[i], epsilon_value * Variance_n[i]) for i in range(len(Variance_N))]

  N_parameter_noreshape = [tf.random.normal(tf.shape(n_parameter_noreshape[i]), n_parameter_noreshape[i], epsilon_value * Variance_N[i]) for i in
                           range(len(Variance_N))]

  N_parameter = []
  n_parameter = []
  for i in range(len(Variance_N)):
    n_parameter.append(tf.reshape(n_parameter_noreshape[i], tf.shape(theta_G[i])))
    N_parameter.append(tf.reshape(N_parameter_noreshape[i], tf.shape(theta_G[i])))

  G_N_sample = generator_nN(X, M, N_parameter)
  G_n_sample = generator_nN(X, M, n_parameter)

  n_RMSE_loss = tf.sqrt(tf.reduce_mean((M * X - M * G_n_sample) ** 2) / tf.reduce_mean(M))
  N_RMSE_loss = tf.sqrt(tf.reduce_mean((M * X - M * G_N_sample) ** 2) / tf.reduce_mean(M))



  abs_N_n = tf.abs(G_N_sample - G_n_sample)

  thre = tf.constant(value=thre_value, shape=[batch_size, dim])
  # less = tf.reduce_mean(M * tf.cast(tf.less(abs_N_n, thre), tf.float32)) / tf.reduce_mean(M)
  less = tf.reduce_sum(M * tf.cast(tf.less(abs_N_n, thre), tf.float32))

  ## GAIN solver
  D_solver = tf.compat.v1.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
  G_solver = tf.compat.v1.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

  clip_d_op = [var.assign(tf.clip_by_value(var, CLIP[0], CLIP[1])) for var in theta_D]

  ## Iterations
  sess = tf.compat.v1.Session()
  sess.run(tf.compat.v1.global_variables_initializer())

  # Start Iterations

  # Mini-batch generation
  def sample_idx(m, n):
    A = np.random.permutation(m)
    idx = A[:n]
    return idx

  # num = 0
  # pbar = InitBar()
  # for it in tqdm(range(epoch)):
  #   data_list = sample_idx(len(norm_data_x), len(norm_data_x))  # Mini batch
  #   mb_idx_list = []
  #   for i in range(0, len(data_list), batch_size):
  #     if i + batch_size > len(data_list):
  #       break
  #     mb_idx_list.append(data_list[i:i + batch_size])
  #   for mb_idx in mb_idx_list:
  #     num += 1
  #     pbar(num / (epoch * len(mb_idx_list)) * 100)
  #
  #     X_mb = norm_data_x[mb_idx, :]
  #     M_mb = data_m[mb_idx, :]
  #     # Sample random vectors
  #     Z_mb = uniform_sampler(0, 0.01, batch_size, dim)
  #     # Sample hint vectors
  #     H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
  #     H_mb = M_mb * H_mb_temp
  #     # Combine random vectors with observed vectors
  #     X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb
  #     _, D_loss_curr = sess.run([D_solver, D_loss_temp], feed_dict={M: M_mb, X: X_mb, H: H_mb})
  #     sess.run(clip_d_op)
  #     _, G_loss_curr, MSE_loss_curr, G_sample_ = sess.run([G_solver, G_loss_temp, MSE_loss, G_sample], feed_dict={X: X_mb, M: M_mb, H: H_mb})
  #
  # ## Choose suitable number
  # start_search = time.clock()
  #
  # k = 20
  # up_number = len(all_data)
  # down_number = Initial_num
  # median_number = int((up_number + down_number) / 2)
  #
  # data_list = sample_idx(len(data_m_val), len(data_m_val))
  # mb_idx_list = []
  # batch_size_search = 128
  # for i in range(0, len(data_list), batch_size_search):
  #   if i + batch_size_search > len(data_list):
  #     break
  #   mb_idx_list.append(data_list[i:i + batch_size_search])
  #
  # while median_number != down_number and median_number != up_number:
  #   predict_within_num = 0
  #   for w in range(k):
  #     loss_N = 0
  #     loss_n = 0
  #
  #     # # For small dataset
  #
  #     # For large-scale dataset
  #     for mb_idx in mb_idx_list:
  #       M_mb = data_m_val[mb_idx, :]
  #       X_mb = val_norm_data_x[mb_idx, :]
  #       Z_mb = uniform_sampler(0, 0.01, len(mb_idx), dim)
  #       X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb
  #       H_mb_temp = binary_sampler(hint_rate, len(mb_idx), dim)
  #       H_mb = M_mb * H_mb_temp
  #       G_n_sample_, G_N_sample_, N_RMSE_loss_, n_RMSE_loss_, less_ = sess.run([G_n_sample, G_N_sample, N_RMSE_loss, n_RMSE_loss, less],
  #                                                     feed_dict={n_num: median_number, X: X_mb, M: M_mb, H: H_mb})
  #       loss_N += N_RMSE_loss_
  #       loss_n += n_RMSE_loss_
  #
  #     if abs(loss_n - loss_N) < thre_value:
  #       print('loss_N: ', loss_N, '  loss_n:', loss_n, 'Delete: ', abs(loss_n - loss_N), 'predict_within_num:', predict_within_num)
  #       predict_within_num += 1
  #   if predict_within_num > guarantee * k:
  #     up_number = median_number
  #   else:
  #     down_number = median_number
  #   median_number = int((up_number + down_number) / 2)
  #   print("Iterative: Median_number:", median_number,' Guarantee: ', predict_within_num)
  # n_number = 65000
  #
  # end_search = time.clock()
  #
  # print()
  #
  # # print(loss_dic)
  # print("Sample size estimation:", n_number, ' Time: ', end_search - start_search)

  # Retraining with "n" number
  train_list = np.random.randint(len(all_data), size=len(all_data))
  train_data_x = all_data[train_list]
  train_data_m = all_data_m[train_list]

  train_norm_data, train_norm_parameters = normalization(train_data_x)
  train_norm_data_x = np.nan_to_num(train_norm_data, 0)

  pbar = InitBar()
  num = 0
  for it in tqdm(range(epoch)):
    data_list = sample_idx(len(train_data_x), len(train_data_x))  # Mini batch
    mb_idx_list = []
    for i in range(0, len(data_list), batch_size):
      if i + batch_size > len(data_list):
        break
      mb_idx_list.append(data_list[i:i + batch_size])
    for mb_idx in mb_idx_list:
      num += 1
      pbar(num / (epoch * len(mb_idx_list)) * 100)

      X_mb = train_norm_data_x[mb_idx, :]
      M_mb = train_data_m[mb_idx, :]
      # Sample random vectors
      Z_mb = uniform_sampler(0, 0.01, batch_size, dim)
      # Sample hint vectors
      H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
      H_mb = M_mb * H_mb_temp
      # Combine random vectors with observed vectors
      X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb
      _, D_loss_curr = sess.run([D_solver, D_loss_temp], feed_dict={M: M_mb, X: X_mb, H: H_mb})
      sess.run(clip_d_op)
      _, G_loss_curr, MSE_loss_curr = sess.run([G_solver, G_loss_temp, MSE_loss],
                                               feed_dict={X: X_mb, M: M_mb, H: H_mb})

  end = time.clock()
  ##  Final Test
  Z_mb = uniform_sampler(0, 0.01, len(all_data_original), dim)
  M_mb = all_data_original_m
  X_mb = all_norm_original_data_x
  X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

  # all_data_original_m = 1 - np.isnan(all_data_original)
  # all_norm_original_data, all_norm_original_parameters = normalization(all_data_original)
  # all_norm_original_data_x = np.nan_to_num(all_norm_original_data, 0)

  imputed_data = sess.run([G_sample], feed_dict={X: X_mb, M: M_mb})[0]

  imputed_data = all_data_original_m * all_norm_original_data_x + (1 - all_data_original_m) * imputed_data

  # Renormalization
  imputed_data = renormalization(imputed_data, all_norm_original_parameters)

  # Rounding
  imputed_data = rounding(imputed_data, all_data_original)


  print('ALL-time', end - start)
  # SSE_time = end_search - start_search
  # return imputed_data, n_number, SSE_time, end - start
  return imputed_data

def main(data, args):
  # data_name = args.data_name
  # miss_rate = args.miss_rate
  
  gain_parameters = {'batch_size': args.batch_size,
                     'hint_rate': args.hint_rate,
                     'alpha': args.alpha,
                     'iterations': args.iterations,
                     'epoch': args.epoch,
                     'guarantee': args.guarantee}
  
  # Load data and introduce missingness
  # ori_data_x, miss_data_x, data_m = data_loader(data_name, miss_rate)

  start = time.clock()
  imputed_data_x = gain(data, gain_parameters, args.initial_value, args.thre_value, args.epsilon, args.value, args.s_miss)
  end = time.clock()

  # Report the RMSE performance
  # rmse = rmse_loss(ori_data_x, imputed_data_x, data_m)
  # mae = mae_loss(ori_data_x, imputed_data_x, data_m)

  # print('RMSE Performance: ' + str(np.round(rmse, 4)))
  
  return imputed_data_x

# if __name__ == '__main__':
#   # dataset_list = ['Hospital', 'Mobility', 'Weather', 'Search']  # Hospital, Mobility, Weather, Search
#   dataset_list = ['Weather']  # Mobility, Weather, Surveillance, Search, Wine
#   miss_list = [0.2]  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#   # sample_rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#   initial_list = [20000]
#   missing_mechanism = ['MCAR']
#   thre_value = 0.001
#   epsilon = 1.4

#   # for p_miss in miss_list:
#   for initial_num in initial_list:
#     p_miss = 0.2
#     value = 2
#     s_miss = 1
#     for dataset in dataset_list:
#       RMSE_LIST = []
#       MAE_LIST = []
#       time_list = []
#       number_rate = []
#       SSE_time_list = []

#       # All_times = 1
#       # for times in range(All_times):
#       # Inputs for the main function
#       parser = argparse.ArgumentParser()
#       parser.add_argument('--data_name', default=dataset, type=str)
#       parser.add_argument('--miss_rate', help='missing data probability', default=p_miss, type=float)
#       parser.add_argument('--batch_size', help='the number of samples in mini-batch', default=128, type=int)
#       parser.add_argument('--hint_rate',  help='hint probability', default=0.9, type=float)
#       parser.add_argument('--alpha', help='hyperparameter', default=10, type=float)
#       parser.add_argument('--iterations', help='number of training interations', default=50, type=int)
#       parser.add_argument('--epoch', help='number of training epoch', default=100, type=int)
#       parser.add_argument('--guarantee', default=0.95, type=float)
#         #args = parser.parse_args()
#       args = parser.parse_known_args()[0]

#       # Calls main function

#       imputed_data, rmse, mae, time_, n_number, SSE_time = main(args, thre_value, initial_num, epsilon, value, s_miss)

#       print(imputed_data)

