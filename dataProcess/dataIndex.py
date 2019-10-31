# -*- coding: utf-8 -*-
# @Time : 6/21/19 12:17 PM
# @Author : Bei Wu
# @Site : 
# @File : dataIndex.py
# @Software: PyCharm
import json
import time

import langid
import numpy as np

np.set_printoptions(threshold=3000000)

SUBG_NUM = 129574
G_NUM = 6746
SUBC_NUM = 627
C_NUM = 123
SUBS_NUM = 27
SEC_NUM = 8

TRAIN_SIZE = 1350000
VALIDATION_SIZE = 20000


def rmDup_test(test_file):
    # input
    infile = open(test_file, 'r', encoding='utf-8')
    lines_seen = set()
    n = 0
    for line in infile.readlines():
        lineTuple = langid.classify(line)
        if lineTuple[0] == 'en':  # 判断是否为英语数据
            if line not in lines_seen:  # 判断是否为非重复数据
                lines_seen.add(line)
                # print(line)
                line = line.strip()
                line = line.strip('\n')
                line = json.loads(line)
                print(line['subject'])
                print(line['abstract'])
                print(line['cpc'])
                n += 1
    print('***************************************')
    print('The number of instances is: ' + str(n))
    print('***************************************')
    infile.close()


# 删除文本中所有的非英文以及重复数据
def rmDup(input_path, subject_path, abstract_path, label_path):
    # input
    infile = open(input_path, 'r', encoding='utf-8')

    # output: subject-subject, abstract-abstract, cpc-label
    subject = open(subject_path, 'a+')
    abstract = open(abstract_path, 'a+')
    label = open(label_path, 'a+')

    lines_seen = set()
    n = 0
    for line in infile.readlines():
        lineTuple = langid.classify(line)
        if lineTuple[0] == 'en':  # 判断是否为英语数据
            if line not in lines_seen:  # 判断是否为非重复数据
                lines_seen.add(line)
                line = line.strip()
                line = line.strip('\n')
                line = json.loads(line)
                subject.write(str(line['subject']))
                abstract.write(str(line['abstract']))
                label.write(str(line['cpc']))
                subject.write('\n')
                abstract.write('\n')
                label.write('\n')
                n += 1
    print('***************************************')
    print('The number of instances is: ' + str(n))
    print('***************************************')
    infile.close()
    abstract.flush()
    abstract.close()
    label.flush()
    label.close()
    subject.flush()
    subject.close()


# 删除文本中所有的非英文以及重复数据
def rmDup2(input_path, subject_path, abstract_path, label_path):
    # input
    infile = open(input_path, 'r', encoding='utf-8')

    # output: subject-subject, abstract-abstract, cpc-label
    subject = open(subject_path, 'a+')
    abstract = open(abstract_path, 'a+')
    label = open(label_path, 'a+')

    lines_seen = set()
    n = 0
    for line in infile.readlines():
        lines_seen.add(line)
        line = line.strip()
        line = line.strip('\n')
        line = json.loads(line)
        subject_str = str(line['subject'])
        abstract_str = str(line['abstract'])
        cpc_str = str(line['cpc'])
        lineTuple = langid.classify(abstract_str)
        if lineTuple[0] == 'en':  # 判断是否为英语数据
            if subject_str not in lines_seen:  # 判断是否为非重复数据
                lines_seen.add(subject_str)
                subject.write(subject_str)
                abstract.write(abstract_str)
                label.write(cpc_str)
                subject.write('\n')
                abstract.write('\n')
                label.write('\n')
                n += 1
    print('***************************************')
    print('The number of instances is: ' + str(n))
    print('***************************************')
    infile.close()
    abstract.flush()
    abstract.close()
    label.flush()
    label.close()
    subject.flush()
    subject.close()


# 对cpc类别进行进一步的处理
def cpcProcess(label_path):
    label_outpath = str(label_path).replace('firPro', 'secPro')
    labels = open(label_path)
    out_labels = open(label_outpath, 'a+')
    for label in labels:
        label = label.strip('\n').strip('[').strip(']').replace('\'', '')
        out_labels.write(label)
        out_labels.write('\n')
    labels.close()
    out_labels.flush()
    out_labels.close()


# 对cpc类别字符串进行切分，获取section, subsection, class, subclass, (main) group, subgroup
def cpcSeg(label_path):
    labels = open(label_path)
    section_path = str(label_path).replace('label.txt', 'section.txt')
    subsection_path = str(label_path).replace('label.txt', 'subsection.txt')
    class_path = str(label_path).replace('label.txt', 'class.txt')
    subclass_path = str(label_path).replace('label.txt', 'subclass.txt')
    group_path = str(label_path).replace('label.txt', 'group.txt')
    subgroup_path = str(label_path).replace('label.txt', 'subgroup.txt')
    sections = open(section_path, 'a+')
    subsections = open(subsection_path, 'a+')
    classes = open(class_path, 'a+')
    subclasses = open(subclass_path, 'a+')
    groups = open(group_path, 'a+')
    subgroups = open(subgroup_path, 'a+')
    for label in labels:
        sec_list = []
        subs_list = []
        c_list = []
        subc_list = []
        g_list = []
        subg_list = []
        arr = label.strip('\n').split(', ')
        for i in range(len(arr)):
            sec_list.append(arr[i][0])
            subs_list.append(arr[i][:2])
            c_list.append(arr[i][:3])
            subc_list.append(arr[i][:4])
            g_list.append((arr[i].split('-'))[0])
            subg_list.append(arr[i])
        # sections.write(str(list(set(sec_list))).strip('[').strip(']').replace('\'', ''))
        sections.write(str(rmDupCPC(sec_list)).strip('\n').strip('[').strip(']').replace('\'', ''))
        sections.write('\n')
        subsections.write(str(rmDupCPC(subs_list)).strip('\n').strip('[').strip(']').replace('\'', ''))
        subsections.write('\n')
        classes.write(str(rmDupCPC(c_list)).strip('\n').strip('[').strip(']').replace('\'', ''))
        classes.write('\n')
        subclasses.write(str(rmDupCPC(subc_list)).strip('\n').strip('[').strip(']').replace('\'', ''))
        subclasses.write('\n')
        groups.write(str(rmDupCPC(g_list)).strip('\n').strip('[').strip(']').replace('\'', ''))
        groups.write('\n')
        subgroups.write(str(rmDupCPC(subg_list)).strip('\n').strip('[').strip(']').replace('\'', ''))
        subgroups.write('\n')
        sec_list.clear()
        subs_list.clear()
        c_list.clear()
        subc_list.clear()
        g_list.clear()
        subg_list.clear()
    labels.close()
    sections.flush()
    sections.close()
    subsections.flush()
    subsections.close()
    classes.flush()
    classes.close()
    subclasses.flush()
    subclasses.close()
    groups.flush()
    groups.close()
    subgroups.flush()
    subgroups.close()


# 删除list重复字符串元素
def rmDupCPC(list1):
    list2 = []
    for i in list1:
        if i not in list2:
            list2.append(i)
    return list2


# 获取section label为1的专利数据
def singleSec():
    abs_path = '/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/firPro/abstract.txt'
    subject_path = '/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/firPro/subject.txt'
    section_path = '/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/secPro/section.txt'
    subs_path = '/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/secPro/subsection.txt'
    class_path = '/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/secPro/class.txt'
    subc_path = '/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/secPro/subclass.txt'
    group_path = '/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/secPro/group.txt'
    subg_path = '/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/secPro/subgroup.txt'
    abstract = open(abs_path)
    subjects = open(subject_path)
    sections = open(section_path)
    subsections = open(subs_path)
    classes = open(class_path)
    subclasses = open(subc_path)
    groups = open(group_path)
    subgroups = open(subg_path)

    abs_out = str(abs_path).replace('firPro', 'thiPro')
    subjects_out = str(subject_path).replace('firPro', 'thiPro')
    section_out = str(section_path).replace('secPro', 'thiPro')
    subs_out = str(subs_path).replace('secPro', 'thiPro')
    class_out = str(class_path).replace('secPro', 'thiPro')
    subc_out = str(subc_path).replace('secPro', 'thiPro')
    group_out = str(group_path).replace('secPro', 'thiPro')
    subg_out = str(subg_path).replace('secPro', 'thiPro')
    re_abs = open(abs_out, 'a+')
    re_subjs = open(subjects_out, 'a+')
    re_secs = open(section_out, 'a+')
    re_subss = open(subs_out, 'a+')
    re_clas = open(class_out, 'a+')
    re_subclas = open(subc_out, 'a+')
    re_gros = open(group_out, 'a+')
    re_subgros = open(subg_out, 'a+')

    n = 0
    for section, subsection, cla, subclass, group, subgroup, abstract, subject \
            in zip(sections, subsections, classes, subclasses, groups, subgroups, abstract, subjects):
        if not str(section).replace('\n', ' ').isspace():
            if len(str(section).strip('\n').split(', ')) == 1:
                re_abs.write(str(abstract))
                re_subjs.write(str(subject))
                re_secs.write(str(section))
                # re_secs.write('\n')
                re_subss.write(str(subsection))
                re_clas.write(str(cla))
                re_subclas.write(str(subclass))
                re_gros.write(str(group))
                re_subgros.write(str(subgroup))
                n += 1
    print('***************************************')
    print('The number of instances is: ' + str(n))
    print('***************************************')
    abstract.close()
    subjects.close()
    sections.close()
    subsections.close()
    classes.close()
    subclasses.close()
    groups.close()
    subgroups.close()
    re_abs.flush()
    re_abs.close()
    re_subjs.flush()
    re_subjs.close()
    re_secs.flush()
    re_secs.close()
    re_subss.flush()
    re_subss.close()
    re_clas.flush()
    re_clas.close()
    re_subclas.flush()
    re_subclas.close()
    re_gros.flush()
    re_gros.close()
    re_subgros.flush()
    re_subgros.close()


# 统计文本中的非重复标签数量
def statistics(file_path):
    file = open(file_path)
    li = []
    for line in file:
        cpc = str(line).strip('\n').strip(', ').split(', ')
        for i in range(len(cpc)):
            if cpc[i] not in li:
                li.append(cpc[i])
    print('***************************************')
    print('The number of elements is: ' + str(len(li)))
    print('***************************************')


# def prepareLabel():
def to_one_hot(y, n_class):
    return np.eye(n_class)[y]


# convert class labels from words to 0-1 vectors.
def to_vector(in_path, out_path, dict_path):
    label_dict = to_dict(in_path, dict_path)
    print("The lenght of dict is: " + str(len(label_dict)))
    in_file = open(in_path)
    out_file = open(out_path, 'a+')
    for line in in_file:
        vec = np.zeros(len(label_dict)).astype(int)
        cpc = str(line).replace('\n', '').strip(', ').split(', ')
        for i in range(len(cpc)):
            if len(cpc[i].strip()) != 0:
                vec.itemset(label_dict.index(cpc[i].strip()), 1)
        out_file.write(str(vec).replace('\n', '').strip('[').strip(']'))
        out_file.write('\n')
    in_file.close()
    out_file.flush()
    out_file.close()


# convert citations' urls into indexes according to dict
def get_index(in_path, out_path, dict_path):
    dict_file = open(dict_path)
    url_dict = []
    for line1 in dict_file:
        urls = line1.replace('\n', '').strip(', ').split(', ')
        print(len(urls))
        for i in range(len(urls)):
            url_dict.append(urls[i])
    print(len(url_dict))

    in_file = open(in_path)
    out_file = open(out_path, 'a+')
    for line2 in in_file:
        indexes = []
        cite_urls = line2.replace('\n', '').strip(', ').split(', ')
        for j in range(len(cite_urls)):
            if len(cite_urls[j].strip()) != 0:
                indexes.append(url_dict.index(cite_urls[j].strip()))
        print(str(indexes))
        for k in range(len(indexes)):
            out_file.write(str(indexes[k]))
            out_file.write(' ')
        out_file.write('\n')

    dict_file.close()
    in_file.close()
    out_file.flush()
    out_file.close()


# get the dict of labels
def to_dict(in_path, out_path):
    in_file = open(in_path)
    out_file = open(out_path, 'a+')
    label_dict = []
    for line in in_file:
        # cpc = str(line).strip('\n').split(', ')
        cpc = str(line).replace('\n', '').strip(', ').split(', ')
        for i in range(len(cpc)):
            if cpc[i] not in label_dict:
                label_dict.append(cpc[i])
    label_dict = sorted(label_dict)
    out_file.write(str(label_dict).strip('[').strip(']').replace('\'', ''))
    out_file.write('\n')
    in_file.close()
    out_file.flush()
    out_file.close()
    return label_dict


def filling(input_path, output_path):
    citation_file = open(input_path)
    out_file = open(output_path, 'a+')
    for line in citation_file:
        # citation = citation.strip().replace('\n', '')
        if line == '\n':
            citation = np.zeros(128)
            out_file.write(str(citation).strip().replace('[', '').replace(']', '').replace('\n', ''))
        else:
            citation = line.strip().replace('\n', '')
            out_file.write(citation)
        out_file.write('\n')
    citation_file.close()
    out_file.flush()
    out_file.close()


def mergecitation(input_path1, input_path2, output_path):
    pri_feature_file = open(input_path1)
    citation_file = open(input_path2)
    out_file = open(output_path, 'a+')
    for pri_feature, citation in zip(pri_feature_file, citation_file):
        feature = str(pri_feature).strip('\n') + ' ' + str(citation).strip('\n')
        out_file.write(feature + '\n')
    pri_feature_file.close()
    citation_file.close()
    out_file.flush()
    out_file.close()


def merge(pat_path, l_path, o_path):
    abstracts = open(pat_path)
    subsections = open(l_path + 'subsection.txt')
    classes = open(l_path + 'class.txt')
    subclasses = open(l_path + 'subclass.txt')
    groups = open(l_path + 'group.txt')
    out_file = open(o_path, 'a+')
    for abstract, subsection, cls, subclass, group in zip(abstracts, subsections, classes, subclasses, groups):
        feature = str(abstract).strip('\n') + str(subsection).strip('\n') + ' ' \
                  + str(cls).strip('\n') + ' ' + str(subclass).strip('\n') + ' ' + str(group).strip('\n')
        out_file.write(feature + '\n')
    abstracts.flush()
    abstracts.close()
    subsections.flush()
    subsections.close()
    classes.flush()
    classes.close()
    subclasses.flush()
    subclasses.close()
    groups.flush()
    groups.close()
    out_file.flush()
    out_file.close()


def mergefull(pat_path, l_path, cit_path, o_path):
    abstracts = open(pat_path)
    subsections = open(l_path + 'subsection.txt')
    classes = open(l_path + 'class.txt')
    subclasses = open(l_path + 'subclass.txt')
    groups = open(l_path + 'group.txt')
    citations = open(cit_path)
    out_file = open(o_path, 'a+')
    patent_num = 0
    for abstract, subsection, cls, subclass, group, citation in \
            zip(abstracts, subsections, classes, subclasses, groups, citations):
        # if patent_num < TRAIN_SIZE+VALIDATION_SIZE:
        if patent_num < TRAIN_SIZE:
            feature = str(abstract).strip('\n') + str(subsection).strip('\n') + ' ' \
                      + str(cls).strip('\n') + ' ' + str(subclass).strip('\n') + ' ' \
                      + str(group).strip('\n') + ' ' + str(citation).strip('\n')
        else:
            feature = str(abstract).strip('\n') \
                      + str(np.ones(SUBS_NUM)).strip().replace('[', '').replace(']', '').replace('\n', '') + ' ' \
                      + str(np.ones(C_NUM)).strip().replace('[', '').replace(']', '').replace('\n', '') + ' ' \
                      + str(np.ones(SUBC_NUM)).strip().replace('[', '').replace(']', '').replace('\n', '') + ' ' \
                      + str(np.ones(G_NUM)).strip().replace('[', '').replace(']', '').replace('\n', '') + ' ' \
                      + str(citation).strip('\n')
        out_file.write(feature + '\n')
        patent_num += 1
    abstracts.flush()
    abstracts.close()
    subsections.flush()
    subsections.close()
    classes.flush()
    classes.close()
    subclasses.flush()
    subclasses.close()
    groups.flush()
    groups.close()
    citations.flush()
    citations.close()
    out_file.flush()
    out_file.close()


def extract_data(filename, num_patents):
    """Extract the images into a 4D tensor [image index, y, x, channels]."""
    print('Extracting', filename)
    data = np.loadtxt(filename)  # 从文件读取数据，存为numpy数组
    # data = np.frombuffer(data).astype(np.int32)  # 改变数组元素变为int32类型
    data = np.frombuffer(data).astype(np.float32)  # 改变数组元素变为float32类型
    data = data.reshape(num_patents, 100 + SUBS_NUM + C_NUM + SUBC_NUM + G_NUM + 128)  # 所有元素
    return data


if __name__ == '__main__':
    patent_path = '/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/data/abstract_100.txt'
    labels_path = '/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/forPro/'
    citation_path = '/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/citation/citesEmbedding/citation_embedding_full3.txt'
    out_path = '/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/data/feature_citation/full2_100.txt'
    n = 1385406
    # # merge(patent_path, labels_path, out_path)
    mergefull(patent_path, labels_path, citation_path, out_path)

    out_path_feature_npy = '/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/data/feature_citation/full2_100.npy'

    start_time = time.time()
    data_all = extract_data(out_path, n)
    features_file = np.save(out_path_feature_npy, data_all)
    # labels_file = np.save("/data/users/lzh/bwu/data/LOP/FirstStep-2/lstm/label.npy", labels_all)
    elapsed_time = time.time() - start_time
    print(elapsed_time, 's')

    # in_path = '/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/thiPro/group.txt'
    # out_path = '/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/forPro/group.txt'
    # dict_out_path = '/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/forPro/group_dict.txt'
    # to_vector(in_path, out_path, dict_out_path)
    # inputPath = '/data/users/lzh/bwu/data/LOP/Origin/LOP-All-03.json'
    # subjectPath = '/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/firPro/subject.txt'
    # abstractPath = '/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/firPro/abstract.txt'
    # labelPath = '/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/firPro/label.txt'
    # rmDup(inputPath, subjectPath, abstractPath, labelPath)

    # rmDup_test('/data/users/lzh/bwu/code/patentcls-multiclass/testfile/data.json')
    # cpcProcess(labelPath)

    # cpcSeg('/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/secPro/label.txt')
    # list1 = [11, 22, 11, 22, 33, 44, 55, 55, 66]
    # print(list1)
    # print(list(set(list1)))

    # singleSec()
    # string = 'H'
    # string2 = '\n'
    # strs = string.strip('\n').split(', ')
    # # strs2 = string2.strip('\n')
    # print(str(len(strs)))
    # if not string2.replace('\n', ' ').isspace():
    #     print('hhhhhhhh')
    # if len(strs) == 1:
    #     print('hello world')

    # statistics('/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/thiPro/subgroup.txt')

    # li = ['hello', 'world', 'hi']
    # dict = ['hello', 'world', 'is', 'are', 'hi']
    # num = len(dict)
    # vec = np.zeros(num).astype(int)
    # print(str(vec))
    # for i in range(len(li)):
    #     n = dict.index(li[i])
    #     vec.itemset(n, 1)
    #     print(str(n))
    # print(str(vec))
    # encoder = LabelEncoder()
    # ocean = encoder.fit_transform(li)
    # ocean = np.array([ocean]).T
    # print(ocean)

    # transfer citation features of patents into 0-1 vectors
    # in_filepath = '/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/forPro/citation' \
    #           '/citesPatentPublication/citesPatentPublication.txt'
    # out_filepath = '/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/forPro/citation/citesPatentPublication' \
    #            '/citesPatentPublication_feature.txt'
    # dict_out_path = '/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/forPro/citation/citesPatentPublication' \
    #                 '/citesPatentPublication_dict.txt'
    # statistics(in_path)
    # to_vector(in_filepath, out_filepath, dict_out_path)
    # get_index(in_filepath, out_filepath, dict_out_path)

    # in_filepath = '/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/citation/citesEmbedding/citation_embedding_full2.txt'
    # out_filepath = '/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/citation/citesEmbedding/citation_embedding_full3.txt'
    # filling(in_filepath, out_filepath)

    # mergecitation(out_path, out_filepath, out_path_feature)
