import numpy as np

def softmax(scores):
    # scores: (N, C)
    Shifted = scores - np.max(scores, axis=1, keepdims=True) #stability
    exp_scores = np.exp(Shifted)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs

def cross_entropy_loss(probs, y):
    # probs: (N, C), y: (N,)
    N = probs.shape[0]
    correct_class_probs = probs[np.arange(N), y]
    loss = -np.mean(np.log(correct_class_probs + 1e-120))
    return loss

def softmax_cross_entropy_backward(probs, y):
    # probs: (N, C), y: (N,)
    N, C = probs.shape
    dScores = probs.copy()
    dScores[np.arange(N), y] -= 1
    dScores /= N
    return dScores