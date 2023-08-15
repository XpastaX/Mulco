import json
import settings.ACE2005 as config

train = []
valid = []
test = []


def count_ne(dataset):
    nested_ent_count = 0
    overlapped_ent_count = 0
    nested_sen_count = 0
    overlapped_sen_count = 0
    same_start_count=0
    same_end_count=0
    depth = {}
    ent_len = {}
    for index, item in enumerate(dataset):
        is_nested = False
        is_overlapped = False
        cate_dict = item['label']
        indice_list = []
        # obtain all indices [a,b] of entities
        cate_per_entity = []
        entity_list = []
        for cate in cate_dict:
            entities_of_cate = cate_dict[cate]
            for entity in entities_of_cate:
                indice_list += entities_of_cate[entity]
                cate_per_entity += [cate] * len(entities_of_cate[entity])
                entity_list += [entity] * len(entities_of_cate[entity])

        # NE static
        for i, (start1, end1) in enumerate(indice_list):
            nest = False
            overlap = False
            has_same_start = False
            has_same_end = False
            level = 1
            L = end1 - start1 + 1
            if L not in ent_len:
                ent_len[L] = 0
            ent_len[L] += 1
            for j, (start2, end2) in enumerate(indice_list):
                if start1 == start2 and end1 == end2: continue
                # whether is nested
                if start1==start2: has_same_start=True
                if end1==end2: has_same_end=True
                if (start1 <= start2 and end1 >= end2) or (start2 <= start1 and end2 >= end1):
                    is_nested = True
                    nest = True
                # whether is overlapped
                if start2 < start1 < end2 < end1:
                    is_overlapped = True
                    overlap = True
                if start1 < start2 < end1 < end2:
                    is_overlapped = True
                    overlap = True
                # count depth
                if start1 >= start2 and end1 <= end2:
                    level += 1
            if nest: nested_ent_count += 1
            if overlap: overlapped_ent_count += 1
            if has_same_start:same_start_count+=1
            if has_same_end:same_end_count+=1
            if level not in depth:
                depth[level] = 0
            depth[level] += 1
        if is_nested: nested_sen_count += 1
        if is_overlapped: overlapped_sen_count += 1
    return nested_ent_count, overlapped_ent_count, nested_sen_count, \
           overlapped_sen_count, depth, ent_len,same_start_count,same_end_count


# dataset = 'ChiNesE'

with open(config.DataInfo.data_path['train'], 'r', encoding='utf-8') as file:
    for line in file:
        train.append(json.loads(line))
with open(config.DataInfo.data_path['valid'], 'r', encoding='utf-8') as file:
    for line in file:
        valid.append(json.loads(line))
with open(config.DataInfo.data_path['test'], 'r', encoding='utf-8') as file:
    for line in file:
        test.append(json.loads(line))

num_train_char = 0
num_valid_char = 0
num_test_char = 0

for item in train:
    num_train_char += len(item['text'])
for item in valid:
    num_valid_char += len(item['text'])
for item in test:
    num_test_char += len(item['text'])

num_train_entity = 0
num_valid_entity = 0
num_test_entity = 0

cate_count_train = {}
cate_count_valid = {}
cate_count_test = {}

for item in train:
    for label in item['label']:
        if label not in cate_count_train:
            cate_count_train[label] = 0
            cate_count_valid[label] = 0
            cate_count_test[label] = 0
        for entity in item['label'][label]:
            num_train_entity += len(item['label'][label][entity])
            cate_count_train[label] += len(item['label'][label][entity])
for item in valid:
    for label in item['label']:
        for entity in item['label'][label]:
            num_valid_entity += len(item['label'][label][entity])
            cate_count_valid[label] += len(item['label'][label][entity])
for item in test:
    for label in item['label']:
        for entity in item['label'][label]:
            num_test_entity += len(item['label'][label][entity])
            cate_count_test[label] += len(item['label'][label][entity])

print('Distribution')
for cate in cate_count_train:
    print(cate, cate_count_train[cate], cate_count_valid[cate], cate_count_test[cate])

avg_train_char = round(num_train_char / len(train), 1)
avg_valid_char = round(num_valid_char / len(valid), 1)
avg_test_char = round(num_test_char / len(test), 1)
avg_train_entity = round(num_train_entity / len(train), 1)
avg_valid_entity = round(num_valid_entity / len(valid), 1)
avg_test_entity = round(num_test_entity / len(test), 1)

# find nested NE
train_nne, train_one, train_ns, train_os, train_depth, train_L,train_ss,train_se = count_ne(train)
valid_nne, valid_one, valid_ns, valid_os, valid_depth, valid_L,valid_ss,valid_se = count_ne(valid)
test_nne, test_one, test_ns, test_os, test_depth, test_L,test_ss,test_se = count_ne(test)

print('Static')
table = '\ttrain\tvalid\ttest\n'
table += f"num. sent\t{len(train)}\t{len(valid)}\t{len(test)}\n"
table += f"num. nest\t{train_ns}\t{valid_ns}\t{test_ns}\n"
table += f"num. over\t{train_os}\t{valid_os}\t{test_os}\n"
table += f"num. ent\t{num_train_entity}\t{num_valid_entity}\t{num_test_entity}\n"
table += f"avg. ent\t{avg_train_entity}\t{avg_valid_entity}\t{avg_test_entity}\n"
table += f"num. nest\t{train_nne}\t{valid_nne}\t{test_nne}\n"
table += f"avg. nest\t{train_nne / len(train)}\t{valid_nne / len(valid)}\t{test_nne / len(test)}\n"
table += f"num. over\t{train_one}\t{valid_one}\t{test_one}\n"
table += f"avg. over\t{train_one / len(train)}\t{valid_one / len(valid)}\t{test_one / len(test)}\n"
table += f"num. char\t{num_train_char}\t{num_valid_char}\t{num_test_char}\n"
table += f"avg. char\t{avg_train_char}\t{avg_valid_char}\t{avg_test_char}\n"
print(table)

print(f'train_same_start:{train_ss}\t train_same_end:{train_se}')
print(f'valid_same_start:{valid_ss}\t valid_same_end:{valid_se}')
print(f'test_same_start :{test_ss}\t test_same_end :{test_se}')
print(f'all_same_start :{train_ss+valid_ss+test_ss}\t allt_same_end :{train_se+valid_se+test_se}')

# print depth
max_depth = max(len(train_depth), len(valid_depth), len(test_depth))
for i in range(1, max_depth + 1):
    for d in [train_depth, valid_depth, test_depth]:
        if i not in d:
            d[i] = 0
print('Depth')
for cate in train_depth:
    print(f"{cate}\t{train_depth[cate]}\t{valid_depth[cate]}\t{test_depth[cate]}")

max_L = max(len(train_L), len(valid_L), len(test_L))
for i in range(1, max_L + 1):
    for d in [train_L, valid_L, test_L]:
        if i not in d:
            d[i] = 0

print('Depth')
for cate in range(1, max_L + 1):
    print(f"{cate}\t{train_L[cate]}\t{valid_L[cate]}\t{test_L[cate]}")

