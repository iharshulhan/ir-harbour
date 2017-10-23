"""This module reads files from disk"""

import os


def crawl_files():
    file_dic = []
    dir = os.path.dirname(__file__)
    for root, dirs, files in os.walk(dir + '/../articles'):
        for file in files:
            file_path = os.path.join(root, file)
            file_dic.append({'name': file_path})
    return file_dic


