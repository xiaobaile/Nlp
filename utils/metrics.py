from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, precision_score, recall_score, f1_score


def metrics(logits, labels):
    """
    常用的评价指标。
    :param logits:
    :param labels:
    :return:
    """
    auc = roc_auc_score(labels, logits)
    loss = log_loss(labels, logits)
    acc = accuracy_score(labels, logits)
    precision = precision_score(labels, logits)
    recall = recall_score(labels, logits)
    f1 = f1_score(labels, logits)
    return auc, loss, acc, precision, recall, f1
