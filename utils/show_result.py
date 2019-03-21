#-*-coding:utf8-*-
import load_data
import numpy as np
from collections import Counter

def evaluate_results(result, neg_id):
    total_p, total_g, right, total, total_right = 0, 0, 0, 0, 0
    for p, g in result:
        total += 1
        if g[0] != neg_id:
            total_g += len(g)
     
        if p != neg_id: total_p += 1
        if p in g: total_right += 1
        if p != neg_id and p in g: right += 1
    if total_p == 0: total_p  = 1
    acc = 1.0 * total_right / total
    pre = 1.0 * right / total_p
    rec = 1.0 * right / total_g
    f1 = 2 * pre * rec / (pre + rec + 0.000001)
    out = 'Total Sample:%d, Total Pred: %d, Total Right_pred:%d, Right Pred_evt: %d, Total Event: %d\n' % (total, total_p, total_right, right, total_g)
    out += 'Accurate:%.3f, Precision: %.3f, Recall: %.3f, F1: %.3f' % (acc, pre, rec, f1)
    return out, f1

def evaluate_results_binary(result, neg_id):
    total_p, total_g, right, total, total_right = 0, 0, 0, 0, 0
    for _p, g in result:
        total += len(_p)
        if g[0] != neg_id:
            total_g += len(g)
        for p in _p:
            if p != neg_id: total_p += 1
            if p in g: total_right += 1
            if p != neg_id and p in g: right += 1
    if total_p == 0: total_p  = 1
    acc = 1.0 * total_right / total
    pre = 1.0 * right / total_p
    rec = 1.0 * right / total_g
    f1 = 2 * pre * rec / (pre + rec + 0.000001)
    out = 'Total Sample:%d, Total Pred: %d, Total Right_pred:%d, Right Pred_evt: %d, Total Event: %d\n' % (total, total_p, total_right, right, total_g)
    out += 'Accurate:%.3f, Precision: %.3f, Recall: %.3f, F1: %.3f' % (acc, pre, rec, f1)
    return out, f1



def padding4cnn_test(data, win):
    return np.array([0] * (win - 1) + list(data) + [0] * (win - 1), dtype='int32')

def classify_cluster(model, clusters, wdict, ydict, max_len, area):
    if area == 'tech':
        ner_tool = ner.NER('科技')
    elif area == 'sport':
        ner_tool = ner.NER('体育')
    elif area == 'ent':
        ner_tool = ner.NER('娱乐')
    rydict = {v : k  for k, v in ydict.items()}
    ret = [] 
    ret_debug_info = []
    total_right, total_p_r, total_predict = 0, 0, 0
    for c in clusters:
        tmps = c.strip().split('\t')
        if len(tmps) < 2: continue
        titles = tmps[1].split('##')
        titles = filter(lambda x: len(x.strip()) > 0, titles)
        g_label = tmps[0].strip()

        if g_label != 'negative': total_right += 1

        ans = _classify_cluster(model, c, wdict, ydict, max_len)
        c_ans = zip(titles, [rydict[p] for p in ans])
        out_str = '\n'.join([': '.join(x) for x in c_ans]) + '\n'
        out_str += ('gold-ans: %s, ' % g_label)
        if ans is None or len(ans) == 0:
            ret.append('null')
            out_str += 'predict-ans: null\n'
        else:
            count = Counter(ans)
            sorted_ans = sorted(count.items(), cmp=lambda a, b:cmp(a[1], b[1]), reverse=True)

            if len(sorted_ans) > 1 and sorted_ans[0][1] == sorted_ans[1][1]: 
                out_str += ('predict-ans: negative\n')
            else:
                label = rydict[sorted_ans[0][0]]
                ans.append(label)
                if label != 'negative':
                    total_predict += 1
                    if label == g_label: total_p_r += 1
                out_str += ('predict-ans: %s\n' % label)
        ret_debug_info.append(out_str)

    print 'Total_right: %d, total_predict:%d, predict_right:%d' % (total_right, total_predict, total_p_r)
    p = 1.0 * total_p_r / total_predict
    r = 1.0 * total_p_r / total_right
    f = 2 * p * r / (p + r)
    print 'Precision:%.3f, Recall: %.3f, F1: %.3f' % (p, r, f)  
    return ret, ret_debug_info   
