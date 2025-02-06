import numpy as np


class Metrics(object):
    def __init__(self, args):
        super().__init__()
        self.PAD = 0
        self.k_list = args.metric_k

    def compute_metric(self, y_prob, y_true):
        k_list = self.k_list
        scores_len = 0
        y_prob = np.array(y_prob)
        y_true = np.array(y_true)

        scores = {'hit@' + str(k): [] for k in k_list}
        scores.update({'map@' + str(k): [] for k in k_list})
        for p_, y_ in zip(y_prob, y_true):
                actual_classes = np.where(y_ == 1)[0]
                if len(actual_classes) > 0:
                    scores_len += 1.0
                    p_sort = p_.argsort()
                    for k in k_list:
                        top_k = p_sort[-k:][::-1]
                        # 计算 hit@k
                        hit = any([cls in top_k for cls in actual_classes])
                        scores['hit@' + str(k)].append(1. if hit else 0.)
                        # 计算 map@k
                        scores['map@' + str(k)].append(apk(actual_classes, top_k, k))

        scores = {k: np.mean(v) for k, v in scores.items()}
        return scores, scores_len


def apk(actual, predicted, k):
    """ Computes the average precision at K. """
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / min(len(actual), k)
