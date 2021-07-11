import argparse
import json

dic_path = '.\\data\\agri_chem\\data.json'
dic = {}
with open(dic_path, 'r') as f:
    dic = json.load(f)

def open_dic(path):
    """
    opens data file and saves to global variable dic
    Args:
        path (string): path of data file
    
    """
    global dic
    with open(path, 'r') as f:
        dic = json.load(f)

def update_file():
    """
        write updates to data file
    """
    with open(dic_path, 'w', encoding='utf-8'):
        json.dum(dic, f, indent='\t')

def print_file():
    """
        print entire data file
    """
    print(dic)

def check_dic(text=''):
    """
        check whether text is in dic
        Args:
            text (string): key of dictionary
        Return:
            True if exist, False if doesn't exist
    """

    if text == '꽉자바':
        text = '깍자바'
    if text in dic:
        return True
    return False

def get_summary(text=None):
    """
        Args:
            text (string): key of dictionary, 농약 이름
        Return:
            tipe (string): 농약 종류
            summary (string): 농약 설명
            names (string): 품목명
    """
    if text == '꽉자바':
        text = '깍자바'
    if text == None or text not in dic:
        return None
    
    if text == '꽉자바':
        text = '깍자바'

    tipe = dic[text]['종류']
    summary = dic[text]['설명']
    names = dic[text]['품목명']

    return tipe, summary, names

def insert(text):
    """
        insert to data
        Args:
            text (string): 형식 = '이름 종류 설명 품목명'
    """
    global dic
    title, tipe, summary, names = text.split(' ')
    
    dic[title] = {}
    dic[title]['종류'] = tipe
    dic[title]['설명'] = summary
    dic[title]['품목명'] = names

    update_file()

def remove(text):
    """
        Args:
            text(string): name of chem to be removed from data
    """
    if text not in dic:
        print('doesn\'t exist')
        return
    
    del dic[text]

    update_file()

def remove_all():
    """
        clears data
    """
    global dic
    dic = {}

    update_file()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='.\\data\\agri_chem\\data.json', help='path of json file')
    parser.add_argument('--mode', type=str, choices=['print', 'insert', 'remove', 'delete'], default='print')
    parser.add_argument('--insert', type=str, default='', help='name type summary names')
    parser.add_argument('--remove', type=str, default='', help='name')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    dic_path = opt.path

    if opt.mode == 'print':
        print_file()
    elif opt.mode == 'insert':
        insert(opt.insert)
    elif opt.mode == 'remove':
        remove(opt.remove)
    elif opt.mode == 'delete':
        remove_all()