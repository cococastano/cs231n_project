import numpy as np
from random import shuffle

def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  scores = np.dot(X, W)

  # to prevent numerical instability
  adjusted_scores = scores - np.max(scores, axis=1, keepdims=True)
  probs = np.exp(adjusted_scores) / np.sum(np.exp(adjusted_scores), axis=1, keepdims=True)

  dummy = 1e-14  # just to avoid dividing by zero
  num_train = X.shape[0]
  loss = -np.sum(np.log(probs[np.arange(num_train), y] + dummy)) / num_train

  d_scores = probs.copy()
  d_scores[range(num_train), y] -= 1
  d_scores /= num_train
  dW = np.dot(np.transpose(X), d_scores)

  return loss, dW

