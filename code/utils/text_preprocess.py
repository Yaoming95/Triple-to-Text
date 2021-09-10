import json
import nltk
import re


def chinese_tokenize(s):
    new_s = ""
    cur_state = False
    for uchar in s:
        if uchar >= u'\u4e00' and uchar <= u'\u9fa5' or (uchar in "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+"):
            if cur_state:
                new_s+=" "
            new_s+=(uchar + ' ')
        else:
            cur_state = True
            new_s += uchar
    return new_s

def tokenlizer(value):
    return nltk.word_tokenize(chinese_tokenize(value))

def get_max_len(sentences):
    sentences = list(map(tokenlizer, sentences))
    return max(list(map(len, sentences)))


def load_data(data_path):
    original_text = list()
    delex_text = list()
    original_tuples = list()
    replace_rules = list()
    delex_tuples = list()
    with open(data_path, "r", encoding="UTF-8") as json_data:
        for line in json_data:
            raw_data = json.loads(line)
            original_text.append(raw_data["origin_text"])
            delex_text.append(raw_data["delex_text"])
            original_tuples.append(raw_data["origin_tuples"])
            replace_rules.append(raw_data["replace_rules"])
            delex_tuples.append(raw_data["delex_tuples"])
    return original_text, delex_text, original_tuples, replace_rules, delex_tuples


def process_tuple(tuple_sentence):
    tuple_sentence = tuple_sentence.split(";")
    tuple_list = []
    for tuple in tuple_sentence:
        if tuple == "":
            continue
        tuple = tuple.split(",")
        tuple_list.append(tuple)
    return tuple_list


def init_data(data_path, to_lower = False):
    """
    init data input

    Returns:
        original_text
        delex_text
        original_tuples
        delex_tuples
        replace_rules
        labels: one-hot, optional
    """
    original_text, delex_text, original_tuples, replace_rules, delex_tuples = load_data(data_path)
    if to_lower:
        original_text = list(map(lambda x: x.lower(), original_text))
        delex_text = list(map(lambda x: x.lower(), delex_text))
        original_tuples = list(map(lambda x: x.lower(), original_tuples))
        # replace_rules = list(map(lambda x: x.lower(), replace_rules))
        delex_tuples = list(map(lambda x: x.lower(), delex_tuples))
    return original_text, delex_text, original_tuples, delex_tuples, replace_rules


def relex(sentences, replace_rules):
    for replace_rule in replace_rules:
        sentences = re.sub('\\b' + replace_rule[1] + '\\b', replace_rule[0], sentences)
    return sentences


def relex_list(sentences_list, replace_rules_list):
    return list(map(lambda x, y: relex(x, y), sentences_list, replace_rules_list))


if __name__ == '__main__':
    init_data("../../data/webnlg_debug.txt")
