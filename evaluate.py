from collections import Counter
import os
import json


def get_entity_bio(seq, id2tag):
    # 将一个句子里面的实体全部找出来，并改成[[tag1, startid, endid], [], []]的结构
    chunks = []
    chunk = [-1, -1, -1]

    for index, tag in enumerate(seq):
        #  这里直接使得输入str还是id均可以进行操作，避免了HMM直接返回了str而不是id从而导致了多一步操作
        if not isinstance(tag, str):
            tag = id2tag[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = index
            chunk[0] = tag.split('-')[1]
            chunk[2] = index
            if index == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = index
            if index == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def compute(origin, found, right):
    # origin代表的事实上的实体个数，found代表发现的实体个数，right代表预测正确的个数，这也适用于某一类型的实体
    recall = 0 if origin == 0 else (right / origin)
    precision = 0 if found == 0 else (right / found)
    f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
    return recall, precision, f1


def result(origins, founds, rights):
    class_info = {}
    origin_counter = Counter([x[0] for x in origins])
    found_counter = Counter([x[0] for x in founds])
    right_counter = Counter([x[0] for x in rights])
    for type_, count in origin_counter.items():  # 分类型计算
        origin = count
        found = found_counter.get(type_, 0)
        right = right_counter.get(type_, 0)
        recall, precision, f1 = compute(origin, found, right)
        class_info[type_] = {"acc": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}
    origin = len(origins)
    found = len(founds)
    right = len(rights)
    recall, precision, f1 = compute(origin, found, right)
    return {'acc': precision, 'recall': recall, 'f1': f1}, class_info


def update(tag_paths, pred_paths, id2tag):
    '''
    将类型的列表转换成[tag, start_index, end_index]
    '''
    origins = []
    founds = []
    rights = []
    for tag_path, pre_path in zip(tag_paths, pred_paths):
        tag_entities = get_entity_bio(tag_path, id2tag)  # 正确的实体
        pre_entities = get_entity_bio(pre_path, id2tag)  # 预测的实体
        origins.extend(tag_entities)
        founds.extend(pre_entities)
        rights.extend([pre_entity for pre_entity in pre_entities if pre_entity in tag_entities])  # 判断一个实体标签和位置相同即是预测正确
    return origins, founds, rights


def metric(preds_list, true_preds_list_tag, id2tag):
    origins, founds, rights = update(preds_list, true_preds_list_tag, id2tag)
    info, class_info = result(origins, founds, rights)
    return info, class_info


def output_result(words_preds_list, true_tags_list, is_HMM, id2tag):
    folder = os.path.exists('My output results')  # 创建一个新的文件夹目录放置转换为BIO格式的文件
    if not folder:
        os.makedirs('My output results')
    if is_HMM:
        words_list, preds_list = words_preds_list
        with open('My output results/my_hmm_BIO_results.txt', 'w') as f:
            for words, preds in zip(words_list, preds_list):
                for word, pred in zip(words, preds):
                    f.write('{} {}'.format(word, pred))  # 因为HMM传入的为str形式
                    f.write('\n')
                f.write('\n')  # 每一句的结束需要换行
        data = []
        for words, true_tags, preds in zip(words_list, true_tags_list, preds_list):
            true_entities = get_entity_bio(true_tags, id2tag)
            true_entities_new = [str((true_entity[0], true_entity[1], true_entity[2])) for true_entity in true_entities]
            pred_entities = get_entity_bio(preds, id2tag)
            pred_entities_new = [str((pred_entity[0], pred_entity[1], pred_entity[2])) for pred_entity in pred_entities]
            data.append({"text": " ".join(words), "ent_predict": "; ".join(pred_entities_new), "golen_label":
                "; ".join(true_entities_new)})
        data_json = json.dumps(data, indent=2)
        with open('My output results/my_hmm_Ent_results.json', 'w', newline='\n') as f:
            f.write(data_json)

    else:
        words_list, preds_list = words_preds_list
        with open('My output results/my_bilstm_BIO_results.txt', 'w') as f:
            for words, preds in zip(words_list, preds_list):
                for word, pred in zip(words, preds):
                    f.write('{} {}'.format(word, id2tag[pred]))  # 因为HMM传入的为str形式
                    f.write('\n')
                f.write('\n')  # 每一句的结束需要换行
        data = []
        for words, true_tags, preds in zip(words_list, true_tags_list, preds_list):
            true_entities = get_entity_bio(true_tags, id2tag)
            true_entities_new = [str((true_entity[0], true_entity[1], true_entity[2])) for true_entity in true_entities]
            pred_entities = get_entity_bio(preds, id2tag)
            pred_entities_new = [str((pred_entity[0], pred_entity[1], pred_entity[2])) for pred_entity in pred_entities]
            data.append({"text": " ".join(words), "ent_predict": "; ".join(pred_entities_new), "golen_label":
                "; ".join(true_entities_new)})
        data_json = json.dumps(data, indent=2)
        with open('My output results/my_bilstm_Ent_results.json', 'w', newline='\n') as f:
            f.write(data_json)
