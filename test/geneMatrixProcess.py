# -*- coding: utf-8 -*-
# @Time : 8/13/19
# @Author : Bei Wu
# @Site : 
# @File : geneMatrixProcess.py
# @Software: PyCharm
import pandas as pd
import linecache


def transCol(annotation_path, probematrix_path, out_path):
    annotation_file = open(annotation_path, 'r')
    gene_symbols = []
    # get gene_symbol from probematrix.txt
    for line1 in annotation_file:
        gene_symbol = line1.split('\t')[10]
        gene_symbols.append(gene_symbol)
    print(str(gene_symbols))

    probematrix_file = open(probematrix_path, 'r')
    row_title = probematrix_file.readline().split('\t')
    num_row = len(gene_symbols)
    num_column = len(row_title)
    # print(str(num_row))
    # print(str(num_column))
    matrix = [list() for i in range(num_column)]
    annotation_file.close()
    probematrix_file.close()

    # matrix = [[]]

    probematrix_file_read = open(probematrix_path, 'r')
    num = 0
    # generate new matrix from ID_REF to Gene symbol
    for line2 in probematrix_file_read:
        newline = []
        lines = line2.strip('\n').split('\t')[1:]
        newline.append(gene_symbols[num])
        newline.extend(lines)
        # print(newline)
        matrix[num].extend(newline)
        num += 1

    out_file = open(out_path, 'w')
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            out_file.write(str(matrix[i][j]))
            # print(str(matrix[i][j]))
            out_file.write('\t')
        out_file.write('\n')
    probematrix_file_read.close()
    out_file.close()
    # return matrix


def averageMerge(input_path, output_path):
    frame = pd.read_csv(input_path, sep='\t')
    df = frame.groupby(by='Gene Symbol').mean()
    df.sort_index(ascending=False, inplace=True)
    df.to_csv(output_path)


def readtest():
    # file = open('/data/users/lzh/bwu/code/patentcls-multiclass/testfile/pan/annotation.txt', 'r')
    file = open('/data/users/lzh/bwu/code/patentcls-multiclass/testfile/pan/probematrix.txt', 'r')
    line = file.readline()
    line = line.split('\t')
    print(line[10])
    for i in range(len(line)):
        print(str(i) + ': ' + line[i])


def get_line_context(file_path, line_number):
    return linecache.getline(file_path, line_number).strip()


if __name__ == '__main__':
    # probematrix = '/data/users/lzh/bwu/code/patentcls-multiclass/testfile/pan/probematrix.txt'
    # annotation = '/data/users/lzh/bwu/code/patentcls-multiclass/testfile/pan/annotation.txt'
    # outF1 = '/data/users/lzh/bwu/code/patentcls-multiclass/testfile/pan/test1.csv'
    # outF2 = '/data/users/lzh/bwu/code/patentcls-multiclass/testfile/pan/test2.csv'
    # transCol(annotation, probematrix, outF1)
    # averageMerge(outF1, outF2)
    # readtest()
    # string = "http://data.epo.org/linked-data/id/application/EP/98119426 -0.31822318 -0.7149576 -1.0417128 " \
    #          "-0.24300914 0.74961334 0.112334155 -0.0446767 -0.04998379 0.92246807 -1.1471887 0.22831734 -0.51053715 " \
    #          "-0.029321952 0.18742907 -1.0498638 0.4867062 0.70919275 0.4615556 -0.1869805 1.0406872 -0.41658652 " \
    #          "0.6699931 -0.46145338 -0.6775549 0.1659997 -0.17548144 0.66725576 -0.30392 0.5666712 0.15374365 " \
    #          "-0.64435124 0.46313852 -0.07383311 0.52359027 -0.91189986 0.8286786 0.33213332 -0.5863942 0.38175637 " \
    #          "0.029337829 0.6333234 0.5516566 1.0344638 0.4928212 0.735475 -0.035345312 -0.14120767 1.0608839 " \
    #          "0.1596318 -0.6821732 -0.45386836 -0.28061453 -0.18807861 -0.18768635 0.068636596 -1.0390185 -0.3776247 " \
    #          "0.30753002 0.5349669 0.686382 -0.46629074 0.57102084 -0.5381362 0.13221349 -0.42984807 -0.6650943 " \
    #          "-0.9000847 -1.0993345 -0.68388337 0.8592291 0.82283956 -0.08252126 -0.10174571 -1.2584666 -0.70246184 " \
    #          "-0.15407431 -0.8551508 -0.398602 -0.042272583 0.24215667 0.27042836 0.8358109 -0.10714445 -0.37139943 " \
    #          "0.4372843 0.60958725 0.22281645 -0.46335018 0.12321997 0.63728946 0.30967748 -0.66617346 -0.83439964 " \
    #          "0.09021655 0.44788778 -0.5773898 0.28716403 0.9494494 0.11804763 0.4453329 -0.0853478 -0.07420184 " \
    #          "-0.9037269 0.64409363 -0.61189866 0.058962043 0.8961165 0.41805908 -0.2182993 0.26976815 -0.13921705 " \
    #          "-0.104453646 0.441841 0.7522038 0.61503595 0.2953477 0.89012146 -0.6050324 0.5784923 1.3533406 " \
    #          "-0.6945899 -0.34909964 0.4175644 0.49472588 0.5058359 0.27534634 -0.22023731 -0.24434611"
    # str2 = string.split(' ')[1:]
    # print(str(len(str2)))
    # print(str(str2).replace('[', '').replace(']', '').replace('\'', '').replace(',', ''))
    file_path = "/data/users/lzh/bwu/code/patentcls-multiclass/testfile/t.txt"
    line_number = 6
    print(get_line_context(file_path, line_number))
