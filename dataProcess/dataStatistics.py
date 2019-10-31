# -*- coding: utf-8 -*-

import numpy as np
# from bert_serving.client import BertClient
#
# np.set_printoptions(suppress=True)
# bc = BertClient(ip='localhost', check_version=False, check_length=False)
#
# a = bc.encode(['please give me a zan!'])
# print(a)


def sectionStatistics():
    section_path = '/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/secPro/section.txt'
    sections = open(section_path)
    n1 = 0
    n2 = 0
    n3 = 0
    n4 = 0
    n5 = 0
    n6 = 0
    n7 = 0
    n8 = 0
    for section in sections:
        if not str(section).replace('\n', ' ').isspace():
            if len(str(section).strip('\n').split(', ')) == 1:
                n1 += 1
            if len(str(section).strip('\n').split(', ')) == 2:
                n2 += 1
            if len(str(section).strip('\n').split(', ')) == 3:
                n3 += 1
            if len(str(section).strip('\n').split(', ')) == 4:
                n4 += 1
            if len(str(section).strip('\n').split(', ')) == 5:
                n5 += 1
            if len(str(section).strip('\n').split(', ')) == 6:
                n6 += 1
            if len(str(section).strip('\n').split(', ')) == 7:
                n7 += 1
            if len(str(section).strip('\n').split(', ')) == 8:
                n8 += 1
    print('***************************************')
    print('The count of 1 multi-class instances is:' + str(n1))
    print('The count of 2 multi-class instances is:' + str(n2))
    print('The count of 3 multi-class instances is:' + str(n3))
    print('The count of 4 multi-class instances is:' + str(n4))
    print('The count of 5 multi-class instances is:' + str(n5))
    print('The count of 6 multi-class instances is:' + str(n6))
    print('The count of 7 multi-class instances is:' + str(n7))
    print('The count of 8 multi-class instances is:' + str(n8))
    print('***************************************')
    sections.close()


if __name__ == '__main__':
    sectionStatistics()
