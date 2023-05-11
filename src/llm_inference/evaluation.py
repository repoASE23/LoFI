
import sys
import logging
import torch
import collections
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# def calculate_scores_pt(ground_truths, predictions):
#     total = 0
#     scores = {'EM': [], 'Precision': [], 'Recall': [], 'F1': [], 'Accuracy': []}
#     for y_trues, y_preds in zip(ground_truths, predictions):
#         if len(y_trues) == 0:
#             continue
#         total += 1
#         scores['EM'].append(y_trues==y_preds)
#         scores['Precision'].append(precision_score(y_trues, y_preds, average='micro') if len(y_preds)>0 else 0)
#         scores['Recall'].append(recall_score(y_trues, y_preds, average='micro') if len(y_preds)>0 else 0)
#         scores['F1'].append(f1_score(y_trues, y_preds, average='micro') if len(y_preds)>0 else 0)
#         scores['Accuracy'].append(accuracy_score(y_trues, y_preds) if len(y_preds)>0 else 0)
#     score_dict = {k: round(sum(v) / total * 100, 2) for k, v in scores.items()}
#     score_dict['Total'] = total
#     return score_dict



def calculate_scores_pt(ground_truths, predictions):
    total = 0
    scores = {'EM': [], 'Precision': [], 'Recall': [], 'F1': []}
    for y_trues, y_preds in zip(ground_truths, predictions):
        if len(y_trues) == 0:
            continue
        total += 1
        scores['EM'].append(y_trues==y_preds)
        common = collections.Counter(y_trues) & collections.Counter(y_preds)
        num_same = sum(common.values())
        # if len(y_trues) == 0 or len(y_preds) == 0:
        #     # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        #     scores['F1'].append(int(y_trues == y_preds))
        if num_same == 0:
            precision, recall, f1 = 0, 0, 0
        else:
            precision = 1.0 * num_same / len(y_preds)
            recall = 1.0 * num_same / len(y_trues)
            f1 = (2 * precision * recall) / (precision + recall)
        scores['Precision'].append(precision)
        scores['Recall'].append(recall)
        scores['F1'].append(f1)

    score_dict = {k: round(sum(v) / total * 100, 2) for k, v in scores.items()}
    score_dict['Total'] = total
    return score_dict



# def get_tokens(s):
#     if not s:
#         return []
#     return normalize_answer(s).split()
#
#
# def compute_f1(a_gold, a_pred):
#     # gold_toks = get_tokens(a_gold)
#     # pred_toks = get_tokens(a_pred)
#     common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
#     num_same = sum(common.values())
#     if len(gold_toks) == 0 or len(pred_toks) == 0:
#         # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
#         return int(gold_toks == pred_toks)
#     if num_same == 0:
#         return 0
#     precision = 1.0 * num_same / len(pred_toks)
#     recall = 1.0 * num_same / len(gold_toks)
#     f1 = (2 * precision * recall) / (precision + recall)
#     return f1


def obtain_model_batch_prediction_tokens(source_ids, predictions, ground_truths, start_desc, end_desc, start_param, end_param, desc_idx, param_idx):
    """
      predictions: (batch, 512) predicted label at each position

    """
    outputs = []
    for src, pred, g_t, s_d, e_d, s_p, e_p in zip(source_ids, predictions, ground_truths, start_desc, end_desc, start_param, end_param):
        output = {"true_desc": [], "true_param": [], "pred_desc": [], "pred_param": []}
        # #### desc
        # piece_gt = g_t[s_d: e_d].tolist() if s_d > -1 else []
        piece_gt = src[s_d: e_d].tolist() if s_d > -1 else []
        output['true_desc'] = piece_gt
        # piece_pred = pred[s_d: e_d].tolist() if s_d > -1 else []
        piece_pred = src[pred==desc_idx].tolist()
        output['pred_desc'] = piece_pred

        # #### param
        # piece_gt = g_t[s_p: e_p].tolist() if s_p > -1 else []
        piece_gt = src[s_p: e_p].tolist() if s_d > -1 else []
        output['true_param'] = piece_gt
        # piece_pred = pred[s_p: e_p].tolist() if s_p > -1 else []
        piece_pred = src[pred==param_idx].tolist()
        output['pred_param'] = piece_pred

        outputs.append(output)
    return outputs

