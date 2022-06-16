import re
import numpy as np
import pandas as pd
from collections import defaultdict
import random
import json
import codecs
import numpy
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('limit', type=int)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        if isinstance(obj, numpy.floating):
            return float(obj)
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def spliteKeyWord(str):
    regex = r"[\u4e00-\ufaff]|[0-9]+|[a-zA-Z]+\'*[a-z]*|[^\w\s]"
    matches = re.findall(regex, str, re.UNICODE)
    return matches

def strided_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))

def pattern_index_broadcasting(all_data, search_data):
    n = len(search_data)
    all_data = np.asarray(all_data)
    all_data_2D = strided_app(np.asarray(all_data), n, S=1)
    return np.flatnonzero((all_data_2D == search_data).all(1))

def get_start_end(s,b):
    l = spliteKeyWord(s)
    m = spliteKeyWord(b)
    out = list(pattern_index_broadcasting(l, m)[:,None] + np.arange(len(m)))

    if out != []:
        result = out[0]
        if len(result)==1:
            return [result[0],result[0],'entity'], l
        if len(result)>1:
            return [result[0], result[-1], 'entity'], l
    else:
        return [], l

def test(filename, sheetname):
    df = pd.read_excel(filename, sheet_name=sheetname,dtype={"宾语":"string","主语":"string","产业链":"string"})
    df = df.dropna(how='all')
    df = df.fillna('')
    sentences = df['原句'].to_list()
    subjects = df['主语'].to_list()
    relations = df['关系'].to_list()
    rr = list(set(relations))


def create_json_data(filename, sheetname):
    df = pd.read_excel(filename, sheet_name=sheetname,dtype={"宾语":"string","主语":"string","产业链":"string"})
    df = df.dropna(how='all')
    df = df.fillna('')
    sentences = df['原句'].to_list()
    subjects = df['主语'].to_list()
    relations = df['关系'].to_list()
    rr = list(set(relations))
    objects = df['宾语'].to_list()
    domains = df['产业链'].to_list()
    companies = df['公司别称'].to_list()

    f_companies = []
    current_company = ''
    for i in range(len(companies)):
        if companies[i] != '':
            f_companies.append(companies[i])
            current_company = companies[i]
        else:
            f_companies.append(current_company)

    datas = []
    data = defaultdict(list)
    current_relation = ''

    for i in range(len(sentences)):

        sentence = sentences[i]
        if sentence != '':
            datas.append(data)
            data = defaultdict(list)

            data['text'] = f_companies[i] + ':' + sentence
            current_relation = relations[i]
            if subjects[i] != '':
                data['subject'].append(subjects[i])
            if subjects[i] == '':
                if current_relation == '无关系':
                    data['subject'].append(subjects[i])
                else:
                    data['subject'].append(f_companies[i])


            data['relation'].append(current_relation)
            data['object'].append(objects[i])
            data['domain'].append(domains[i])
        else:
            if subjects[i] != '':
                data['subject'].append(subjects[i])
            else:
                data['subject'].append(f_companies[i])
            data['object'].append(objects[i])
            data['domain'].append(domains[i])
            if relations[i] != '':
                data['relation'].append(relations[i])
            else:
                data['relation'].append(current_relation)
    datas.append(data)
    count = 0

    train_list = []
    for i in range(1, len(datas)):

        d = datas[i]
        dict = {}
        sentence = d['text']
        rt = d['relation']

        if rt[0] == '无关系':
            if count>100:
                continue
            else:
                count+=1

        ners = []

        subjects = list(set(d['subject']))

        subjects = [l for l in subjects if l != '']
        subjects_process = []
        for s in subjects:
            parts = re.split('、|，', s)
            subjects_process += parts
        subjects_process = [s for s in subjects_process if s != '']
        #subjects_ners, sentence_list = get_start_end(sentence, subjects_process)
        objects = list(set(d['object']))
        objects = [l for l in objects if l != '']
        objects_process = []

        for o in objects:
            parts = re.split('、|，', o)
            objects_process += parts
        objects_process = [o for o in objects_process if o != '']
        #objects_ners, sentence_list = get_start_end(sentence, objects_process)
        domains = list(set(d['domain']))
        domains = [l for l in domains if l != '']
        domains_process = []
        for d_ in domains:
            parts = re.split('、|，', d_)
            domains_process += parts
        domains_process = [d for d in domains_process if d != '']
        #domains_ners, sentence_list = get_start_end(sentence, domains_process)
        process_entities = subjects_process + objects_process + domains_process
        process_entities = list(set(process_entities))

        if process_entities == []:
            sentence_list = spliteKeyWord(sentence)

        for entity in process_entities:
            if entity in sentence:
                ner_labeled_entity, sentence_list = get_start_end(sentence, entity)
                ners.append(ner_labeled_entity)

        ners_calculated = []
        for n in ners:
            entity = ''.join(sentence_list[n[0]:n[1]+1])
            ners_calculated.append(entity)

        relations_ner = []
        relations_calculated = []

        subjects = d['subject']
        objects = d['object']
        relations = d['relation']

        for i in range(len(subjects)):
            subject = subjects[i]
            object = objects[i]
            relation = relations[i]

            s_parts = re.split('、|，', subject)
            s_parts = [l for l in s_parts if l != '']

            for s in s_parts:
                if s in sentence:
                    s_e_1, _ = get_start_end(sentence, s)
                    start_1 = s_e_1[0]
                    end_1 = s_e_1[1]
                    parts = re.split('、|，', object)
                    parts = [l for l in parts if l != '']
                    for l in parts:
                        if l in sentence:
                            s_e_2, _ = get_start_end(sentence, l)
                            start_2 = s_e_2[0]
                            end_2 = s_e_2[1]
                            relations_ner.append([start_1, end_1, start_2, end_2, relation])
                            relations_calculated.append([''.join(sentence_list[start_1:end_1+1]), ''.join(sentence_list[start_2:end_2+1]), relation  ])

        domains = d['domain']

        for i in range(len(domains)):
            domain = domains[i]
            object = objects[i]

            if domain == '':
                continue
            else:
                d_parts = re.split('、|，', domain)
                d_parts = [l for l in d_parts if l != '']

                for d in d_parts:
                    if d in sentence:
                        s_e_1, _ = get_start_end(sentence, d)
                        start_1 = s_e_1[0]
                        end_1 = s_e_1[1]

                        parts = re.split('、|，', object)
                        parts = [l for l in parts if l != '']

                        for l in parts:
                            if l in sentence:
                                s_e_2, _ = get_start_end(sentence, l)
                                start_2 = s_e_2[0]
                                end_2 = s_e_2[1]
                                relations_ner.append([start_1, end_1, start_2, end_2, 'domain_relation'])
                                relations_calculated.append(
                                    [''.join(sentence_list[start_1:end_1 + 1]), ''.join(sentence_list[start_2:end_2 + 1]), 'domain_relation'])



        #########################################################################################
        relations_ground = []

        for i in range(len(subjects)):
            subject = subjects[i]
            object = objects[i]
            relation = relations[i]

            s_parts = re.split('、|，', subject)
            s_parts = [l for l in s_parts if l != '']

            for s in s_parts:
                parts = re.split('、|，', object)
                parts = [l for l in parts if l != '']
                for l in parts:
                    relations_ground.append([s,l, relation])


        for i in range(len(domains)):
            domain = domains[i]
            object = objects[i]
            if domain == '':
                continue
            else:
                d_parts = re.split('、|，', domain)
                d_parts = [l for l in d_parts if l != '']

                for d in d_parts:
                    parts = re.split('、|，', object)
                    parts = [l for l in parts if l != '']

                    for l in parts:
                        relations_ground.append([d,l, 'domain_relation'])
        #########################################################################################
        dict['text'] = sentence
        dict['sentences'] = [sentence_list]
        dict['ner'] = [ners]
        dict['relations'] = [relations_ner]
        dict['relations_calculated'] = relations_calculated
        dict['relations_ground'] = relations_ground
        dict['ner_calculated'] = ners_calculated
        dict['ner_ground'] = process_entities

        train_list.append(dict)

    print(count)
    return train_list


if __name__ == '__main__':
    A = parser.parse_args()

    train_list_1 = create_json_data('data/train_negative_1.xlsx','朱泽宇')
    train_list_2 = create_json_data('data/train_negative_2.xlsx', '陈梓铤')

    train_list = train_list_1 + train_list_2
    c = A.limit // 300
    dev_list = train_list[(c - 1) * 300:(c - 1) * 300 + 300]
    print('from {} to {}'.format((c - 1) * 300,(c - 1) * 300 + 300))
    test_list = train_list[(c - 1) * 300:(c - 1) * 300 + 300]


    train_list = [x for x in train_list if x not in dev_list]

    # random.shuffle(train_list)
    #
    # dev_list = train_list[:A.limit]
    # test_list = train_list[:A.limit]
    # train_list = train_list[A.limit:]

    with codecs.open('data/json/train.json', 'w', encoding='utf-8') as fp:
        fp.write(
            '\n'.join(json.dumps(i, ensure_ascii=False,cls=NpEncoder) for i in train_list) +
            '\n')
    with codecs.open('data/json/dev.json', 'w', encoding='utf-8') as fp:
        fp.write(
            '\n'.join(json.dumps(i, ensure_ascii=False,cls=NpEncoder) for i in dev_list) +
            '\n')
    with codecs.open('data/json/test.json', 'w', encoding='utf-8') as fp:
        fp.write(
            '\n'.join(json.dumps(i, ensure_ascii=False,cls=NpEncoder) for i in test_list) +
            '\n')



    # with codecs.open('/tmp/kashif/pytorch-bert-ner/data/json/train.json', 'w', encoding='utf-8') as fp:
    #     fp.write(
    #         '\n'.join(json.dumps(i, ensure_ascii=False,cls=NpEncoder) for i in train_list) +
    #         '\n')
    # with codecs.open('/tmp/kashif/pytorch-bert-ner/data/json/dev.json', 'w', encoding='utf-8') as fp:
    #     fp.write(
    #         '\n'.join(json.dumps(i, ensure_ascii=False,cls=NpEncoder) for i in dev_list) +
    #         '\n')
    # with codecs.open('/tmp/kashif/pytorch-bert-ner/data/json/test.json', 'w', encoding='utf-8') as fp:
    #     fp.write(
    #         '\n'.join(json.dumps(i, ensure_ascii=False,cls=NpEncoder) for i in test_list) +
    #         '\n')




