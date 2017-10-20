"""This module reads files from disk"""

import os


def crawl_files():
    file_dic = []
    for root, dirs, files in os.walk('python-3.6.3-docs-text'):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path) as auto:
                doc = auto.read()
                file_dic.append({'name': file_path, 'doc': doc})
    return file_dic


