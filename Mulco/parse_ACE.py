import json
from transformers import AutoTokenizer
import config
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained(config.plm)

raw_data_path = ['data/ACE2005/raw_train.json', 'data/ACE2005/raw_dev.json', 'data/ACE2005/raw_test.json']

use_cate = ['PER', 'ORG', 'GPE', 'LOC', 'FAC', 'WEA', 'VEH']


def process(__path):
    with open(__path, 'r') as f:
        raw = json.load(f)

    # collect entity categories
    cate = {item: 0 for item in use_cate}
    for item in raw:
        for entity in item['golden-entity-mentions']:
            __cate = entity['entity-type'][:3]
            if __cate in cate:
                cate[__cate] += 1

    print('categories')
    sorted_cate = sorted(cate.items(), key=lambda d: d[0])
    for item in sorted_cate:
        print(item[0], item[1])

    data = []
    for idx, item in enumerate(raw):
        sentence = item['sentence']
        if len(sentence) >= 510:
            continue
        path = item['path']
        label = {}
        ent_cate = {}
        for entity in item['golden-entity-mentions']:
            __cate = entity['entity-type'][:3]
            if __cate not in cate: continue
            __ent = entity['text']
            start = entity['start']
            end = entity['end'] - 1
            # =======================
            # fix bugs
            if end+1 > len(sentence):
                end = len(sentence)-1
            check = '-'.join([__ent,str(start),str(end)])
            if check not in ent_cate:
                ent_cate[check] = __cate
            else: continue
            # =======================
            if __cate not in label:
                label[__cate] = {}
            if __ent not in label[__cate]:
                label[__cate][__ent] = []
            label[__cate][__ent].append([start, end])
        __sample = {'id': idx,
                    'text': sentence,
                    'label': label,
                    'path': path,
                    }
        data.append(__sample)

    # deal with unknown tokens
    count = 0
    for index, sample in tqdm(enumerate(data)):
        sample['id'] = index
        text = list(sample['text'])
        for i, char in enumerate(text):
            if len(tokenizer([char], is_split_into_words=True, add_special_tokens=False)['input_ids']) == 0:
                text[i] = ' '
                count += 1
        sample['text'] = ''.join(text)
    print('Found %s unknown tokens' % count)
    # count the numer of nested entities
    num_nested_child = 0

    num_nested_parent = 0
    nested_id = []
    no_nested_id = []
    nested_category = {key: 0 for key in cate}
    nested_category_child = {key: 0 for key in cate}
    nested_category_parent = {key: 0 for key in cate}

    nested_category_sample = {key: [] for key in cate}
    for index, item in enumerate(data):
        is_nested = False
        cate_dict = item['label']
        indice_list = []
        # obtain all indices [a,b] of entities
        cate_per_entity = []
        for cate in cate_dict:
            entities_of_cate = cate_dict[cate]
            for entity in entities_of_cate:
                indice_list += entities_of_cate[entity]
                cate_per_entity += [cate] * len(entities_of_cate[entity])
        # find nested NE
        for i, (start1, end1) in enumerate(indice_list):
            is_parent = False
            for j, (start2, end2) in enumerate(indice_list):
                if start1 == start2 and end1 == end2:
                    continue
                elif start2 >= start1 and end2 <= end1:
                    # nested
                    num_nested_child += 1
                    is_parent = True
                    is_nested = True
                    cate_child = cate_per_entity[j]
                    nested_category[cate_child] += 1
                    nested_category_child[cate_child] += 1
                else:
                    continue
            if is_parent:
                num_nested_parent += 1
                cate_parent = cate_per_entity[i]
                nested_category_parent[cate_parent] += 1
                nested_category[cate_parent] += 1
                nested_category_sample[cate_parent].append(item['id'])
        if is_nested:
            nested_id.append(index)
        else:
            no_nested_id.append(index)

    num_nested_sample = len(nested_id)
    print(f"In total of {num_nested_sample} nested samples")
    return data


for path in raw_data_path:
    print('================================================================')
    print('processing: ' + path)
    data = process(path)

    with open(path.replace('raw_', ''), 'w', encoding='utf-8') as f:
        for s in data:
            sample = json.dumps(s, ensure_ascii=False) + '\n'
            f.write(sample)
