import glob
import json
import os

from pathlib import Path
from os.path import join

FINISHED_FILES_FOLDER = './dataset/data/finished_files'
ALL_FOLDERS = ['train', 'val', 'test']  


def remove_empty_articles():
    for folder in ALL_FOLDERS:
        print(f"Removing empty articles in folder '{folder.upper()}'.")
        folder_path = join(FINISHED_FILES_FOLDER, folder)

        for article_path in glob.glob(folder_path + '/*.json'):
            with open(article_path, 'r') as article_file:
                article_obj = json.load(article_file)
                if not article_obj['abstract'] or not article_obj['article']:
                    article_number = Path(article_path).stem
                    print(
                        f"Article {article_number} in the {folder} folder either has empty abstract or article body. Deleting it.")
                    os.remove(article_path)

if __name__ == '__main__':
    remove_empty_articles()
