# -*- coding: utf-8 -*-
# @Time : 7/9/19 5:54 PM
# @Author : Bei Wu
# @Site : 
# @File : citationProcess.py
# @Software: PyCharm
import json

import linecache
from pymongo import MongoClient


def operation_mongoDB(in_subject, output_path):
    # input
    subject_file = open(in_subject)

    conn = MongoClient('tdm-p-graph01.fiz-karlsruhe.de', 27017)
    dblist = conn.list_database_names()
    print(dblist)
    databaseName = "patent"
    collectionName = "RelationIndex"
    if databaseName in dblist:
        print(databaseName + " already exists!")
        db = conn.get_database(databaseName)
        myset = db.get_collection(collectionName)
        with open(output_path, 'a+') as f:
            n = 0
            for line in subject_file.readlines():
                subject = line.strip().replace('\n', '')
                for item in myset.find({"subject": subject,
                                        "predicate": "http://data.epo.org/linked-data/def/patent/citationNPL"}):
                    f.write(item["object"])
                    f.write(', ')
                    # print(item["object"])
                f.write('\n')
                n += 1
            print('***************************************')
            print('The number of instances is: ' + str(n))
            print('***************************************')
        # inputfile = open("/data/users/lzh/bwu/code/patentcls-multiclass/testfile/test_mon.txt")
        # for line in inputfile.readlines():
        #     ar = line.strip('\n').split(', ')
        #     for i in range(len(ar)):
        #         print(ar[i])
        # inputfile.close()
    subject_file.close()
    conn.close()


def citesGet(in_subject, input_path, output_path):
    # input
    subject_file = open(in_subject)
    # infile = open(input_path, 'r', encoding='utf-8')

    # output: citation_url
    outfile = open(output_path, 'a+')

    n = 0
    for line in subject_file.readlines():
        print(line)
        subject = str(line).replace('\n', '').strip()
        cites = []
        with open(input_path) as infile:
            for citation in infile.readlines():
                citation = citation.replace('\n', '').strip()
                # print(citation)
                citation = json.loads(citation)
                subject_str = str(citation["subject"]).replace('\n', '').strip()
                # print(subject_str)
                citation_str = str(citation["object"]).replace('\n', '').strip()
                # print(citation_str)
                if subject_str == subject:
                    cites.append(citation_str.strip('\n'))
            print(str(cites))
            outfile.write(str(cites).replace('\n', '').strip('[').strip(']').replace('\'', ''))
            outfile.write('\n')
        n += 1
    print('***************************************')
    print('The number of instances is: ' + str(n))
    print('***************************************')
    subject_file.close()
    outfile.flush()
    outfile.close()


# 只保留最后一个/后的序列号
def reProcess_citationNPL(in_path, out_path):
    input_file = open(in_path)
    out_file = open(out_path, 'a+')
    for line in input_file.readlines():
        urls = line.replace('\n', '').strip(', ').split(', ')
        # print(urls)
        indeces = []
        for i in range(len(urls)):
            # print(urls[i])
            index = urls[i].strip(', ').split('/')[-1]
            indeces.append(index)
        out_file.write(str(indeces).strip('[').strip(']').replace('\'', ''))
        out_file.write('\n')
    input_file.close()
    out_file.flush()
    out_file.close()


def publication2application(in_subject, output_path):
    subject_file = open(in_subject)

    conn = MongoClient('tdm-p-graph01.fiz-karlsruhe.de', 27017)
    dblist = conn.list_database_names()
    print(dblist)
    databaseName = "patent"
    collectionName = "RelationIndex"
    if databaseName in dblist:
        print(databaseName + " already exists!")
        db = conn.get_database(databaseName)
        myset = db.get_collection(collectionName)
        with open(output_path, 'a+') as f:
            n = 0
            for line in subject_file.readlines():
                publication = line.strip().replace('\n', '')
                # for item in myset.find({"subject": subject,
                #                         "predicate": "http://data.epo.org/linked-data/def/patent/application"}):
                #     f.write(item["object"])
                #     f.write(', ')
                #     # print(item["object"])
                for application in myset.find({"subject": publication,
                                               "predicate": "http://data.epo.org/linked-data/def/patent/application"}):
                    f.write(application["object"])
                f.write('\n')
                n += 1
            print('***************************************')
            print('The number of instances is: ' + str(n))
            print('***************************************')
    subject_file.close()
    conn.close()


def citation2Embedding(in_application, in_embedding, out_embedding):
    input_file = open(in_application)
    # input_file_embedding = open(in_embedding)
    # out_file = open(out_embedding, 'a+')
    with open(out_embedding, 'a+') as f:
        for line in input_file.readlines():
            application = line.strip().replace('\n', '')
            with open(in_embedding) as infile:
                for line2 in infile.readlines():
                    embedding = line2.strip().replace('\n', '')
                    if application == embedding.split(' ')[0]:
                        f.write(str(embedding.split(' ')[1:]).replace('[', '').replace(']', '').
                                replace('\'', '').replace(',', ''))
                        f.write('\n')
    input_file.close()
    # input_file_embedding.close()


def get_line_context(file_path, line_number):
    return linecache.getline(file_path, line_number).strip()


def citation2Embedding2(in_application, in_embedding, out_embedding, out_application_null):
    input_file = open(in_application)
    input_file_embedding = open(in_embedding)
    outfile_application_null = open(out_application_null, 'a+')

    item = []
    num1 = 0
    num2 = 0
    for line in input_file_embedding.readlines():
        application1 = line.strip().replace('\n', '').split(' ')[0]
        item.append(application1)
    # print(str(item))
    input_file_embedding.close()

    with open(out_embedding, 'a+') as f:
        for line in input_file.readlines():
            application2 = line.strip().replace('\n', '')
            # print(application2)
            if application2 in item:
                num = item.index(application2)+1
                # print(str(num))
                # print(get_line_context(in_embedding, 2))
                target_line = get_line_context(in_embedding, num)
                # print(target_line)
                embedding = str(target_line.strip().replace('\n', '').split(' ')[1:]).\
                    replace('[', '').replace(']', '').replace('\'', '').replace(',', '')
                f.write(embedding)
                f.write('\n')
                num1 += 1
            else:
                outfile_application_null.write(application2)
                outfile_application_null.write('\n')
                f.write('\n')
                num2 += 1
    print('***************************************')
    print("The number of applications with citation embeddings: " + str(num1))
    print("The number of applications without citation embeddings: " + str(num2))
    print('***************************************')
    input_file.close()
    outfile_application_null.flush()
    outfile_application_null.close()


if __name__ == '__main__':
    # subject_path = '/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/thiPro/subject.txt'
    # in_path = '/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/citation/citesPatentPublication.json'
    # out_path = '/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/forPro/citesPatentPublication.txt'
    # out_path = '/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/forPro/citation/citationNPL/citationNPL.txt'
    # citesGet(subject_path, in_path, out_path)
    # str1 = "http://data.epo.org/linked-data/data/publication/EP/1401236/A1/-".strip('\n')
    # str2 = "http://data.epo.org/linked-data/data/publication/EP/1401236/A1/-".strip('\n')
    # if str1 == str2:
    #     print('True')
    # else:
    #     print('False')
    # subject_path = '/data/users/lzh/bwu/code/patentcls-multiclass/testfile/subject.txt'
    # in_path = '/data/users/lzh/bwu/code/patentcls-multiclass/testfile/cites.json'
    # out_path = '/data/users/lzh/bwu/code/patentcls-multiclass/testfile/t.txt'
    # citesGet(subject_path, in_path, out_path)
    # operation_mongoDB(subject_path, out_path)
    # out_path2 = '/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/forPro/citation/citationNPL/citationNPL2.txt'
    # reProcess_citationNPL(out_path, out_path2)
    application_path = '/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/citation/citesEmbedding/application.txt'
    # publication2application(subject_path, application_path)

    #
    # string = "http://data.epo.org/linked-data/id/application/EP/98119426 -0.31822318 -0.7149576 -1.0417128
    # -0.24300914 0.74961334 0.112334155 -0.0446767 -0.04998379 0.92246807 -1.1471887 0.22831734 -0.51053715
    # -0.029321952 0.18742907 -1.0498638 0.4867062 0.70919275 0.4615556 -0.1869805 1.0406872 -0.41658652 0.6699931
    # -0.46145338 -0.6775549 0.1659997 -0.17548144 0.66725576 -0.30392 0.5666712 0.15374365 -0.64435124 0.46313852
    # -0.07383311 0.52359027 -0.91189986 0.8286786 0.33213332 -0.5863942 0.38175637 0.029337829 0.6333234 0.5516566
    # 1.0344638 0.4928212 0.735475 -0.035345312 -0.14120767 1.0608839 0.1596318 -0.6821732 -0.45386836 -0.28061453
    # -0.18807861 -0.18768635 0.068636596 -1.0390185 -0.3776247 0.30753002 0.5349669 0.686382 -0.46629074 0.57102084
    # -0.5381362 0.13221349 -0.42984807 -0.6650943 -0.9000847 -1.0993345 -0.68388337 0.8592291 0.82283956 -0.08252126
    # -0.10174571 -1.2584666 -0.70246184 -0.15407431 -0.8551508 -0.398602 -0.042272583 0.24215667 0.27042836
    # 0.8358109 -0.10714445 -0.37139943 0.4372843 0.60958725 0.22281645 -0.46335018 0.12321997 0.63728946 0.30967748
    # -0.66617346 -0.83439964 0.09021655 0.44788778 -0.5773898 0.28716403 0.9494494 0.11804763 0.4453329 -0.0853478
    # -0.07420184 -0.9037269 0.64409363 -0.61189866 0.058962043 0.8961165 0.41805908 -0.2182993 0.26976815
    # -0.13921705 -0.104453646 0.441841 0.7522038 0.61503595 0.2953477 0.89012146 -0.6050324 0.5784923 1.3533406
    # -0.6945899 -0.34909964 0.4175644 0.49472588 0.5058359 0.27534634 -0.22023731 -0.24434611" str2 = string.split('
    # ')[1:] print(str(str2).replace('[', '').replace(']', '').replace('\'', '').replace(',', ''))
    embedding_path = '/data/users/lzh/patent/epo/embedding/application_citations_embeddings_order2_epoch50.txt'
    embedding_outpath = \
        '/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/citation/citesEmbedding/citation_embedding_full2.txt'
    embedding_null_outpath = \
        '/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/citation/citesEmbedding/citation_embedding_null2.txt'
    # citation2Embedding(application_path, embedding_path, embedding_outpath)
    # citation2Embedding2('/data/users/lzh/bwu/code/patentcls-multiclass/testfile/application.txt',
    #                     '/data/users/lzh/bwu/code/patentcls-multiclass/testfile/t.txt',
    #                     '/data/users/lzh/bwu/code/patentcls-multiclass/testfile/citation_embedding.txt')
    citation2Embedding2(application_path, embedding_path, embedding_outpath, embedding_null_outpath)
    print('***************************************')
    print("The processing is well done!!!")
    print('***************************************')

