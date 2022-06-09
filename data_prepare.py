# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""duee 1.0 dataset process"""
import os
import sys
import json
from torch.utils.data import Dataset
from utils import read_by_lines, write_by_lines, get_entities

def convert_example_to_feature_for_token_pair(example, tokenizer, label_vocab=None, max_span_len=4, border_vocab=None, max_seq_len=512, no_entity_label="O", ignore_label=-1, is_test=False):
    tokens, labels, border_labels = example
    tokenized_input = tokenizer(
        tokens,
        is_split_into_words=True,
        padding='max_length',
        truncation=True,
        max_length=max_seq_len)

    input_ids = tokenized_input['input_ids']
    token_type_ids = tokenized_input['token_type_ids']
    seq_len = len(tokens) + 2

    if is_test:
        return input_ids, token_type_ids, seq_len
    elif label_vocab is not None:
        labels, border_labels = labels[:(max_seq_len-2)], border_labels[:(max_seq_len-2)]
        encoded_labels = [no_entity_label] + labels + [no_entity_label]
        entity_spans = get_entities(encoded_labels, suffix=False) # trigger spans
        encoded_labels = [label_vocab[x] for x in encoded_labels]

        border_labels = [no_entity_label] + border_labels + [no_entity_label]
        #border_vocab = {'B': 0, 'I':1, 'E': 2, 'O': 3}
        border_labels = [border_vocab[x] for x in border_labels]
        # padding label 
        encoded_labels += [ignore_label] * (max_seq_len - len(encoded_labels))
        border_labels += [ignore_label] * (max_seq_len - len(border_labels))

        
        token_pair_labels = [int(label_vocab[no_entity_label]/2)] * (len(encoded_labels)*max_span_len)
        for span in entity_spans:
            trigger_type, start, end = span
            if end - start >= max_span_len:
                continue
            token_pair_labels[(end-start)*max_seq_len+start] = int(label_vocab['B-'+trigger_type] / 2)
        return input_ids, token_type_ids, seq_len, encoded_labels, border_labels, token_pair_labels

def convert_example_to_feature_multitask_trigger_boundary(example, tokenizer, label_vocab=None, border_vocab=None, max_seq_len=512, no_entity_label="O", ignore_label=-1, is_test=False):
    tokens, labels, border_labels = example
    tokenized_input = tokenizer(
        tokens,
        is_split_into_words=True,
        padding='max_length',
        truncation=True,
        max_length=max_seq_len)

    input_ids = tokenized_input['input_ids']
    token_type_ids = tokenized_input['token_type_ids']
    seq_len = len(tokens) + 2

    if is_test:
        return input_ids, token_type_ids, seq_len
    elif label_vocab is not None:
        labels, border_labels = labels[:(max_seq_len-2)], border_labels[:(max_seq_len-2)]
        encoded_labels = [no_entity_label] + labels + [no_entity_label]
        encoded_labels = [label_vocab[x] for x in encoded_labels]

        border_labels = [no_entity_label] + border_labels + [no_entity_label]
        #border_vocab = {'B': 0, 'I':1, 'E': 2, 'O': 3}
        border_labels = [border_vocab[x] for x in border_labels]
        # padding label 
        encoded_labels += [ignore_label] * (max_seq_len - len(encoded_labels))
        border_labels += [ignore_label] * (max_seq_len - len(border_labels))


        return input_ids, token_type_ids, seq_len, encoded_labels, border_labels

def convert_example_to_feature_multitask_event_type(example, tokenizer, label_vocab=None, event_vocab=None, max_seq_len=512, no_entity_label="O", ignore_label=-1, is_test=False):
    tokens, labels, event_types = example
    tokenized_input = tokenizer(
        tokens,
        is_split_into_words=True,
        padding='max_length',
        truncation=True,
        max_length=max_seq_len)

    input_ids = tokenized_input['input_ids']
    token_type_ids = tokenized_input['token_type_ids']
    seq_len = len(tokens) + 2

    if is_test:
        return input_ids, token_type_ids, seq_len
    assert label_vocab is not None
    labels = labels[:(max_seq_len-2)]
    encoded_labels = [no_entity_label] + labels + [no_entity_label]
    encoded_labels = [label_vocab[x] for x in encoded_labels]

    # padding label 
    encoded_labels += [ignore_label] * (max_seq_len - len(encoded_labels))
    event_labels = [0] * len(event_vocab)
    for et in event_types:
        event_labels[event_vocab[et]] = 1
    


    return input_ids, token_type_ids, seq_len, encoded_labels, event_labels

class DuEventExtraction(Dataset):
    """DuEventExtraction"""
    def __init__(self, data_path):
        
        self.word_ids = []
        self.label_ids = []
        #self.border_label_ids = []
        self.event_types = []
        with open(data_path, 'r', encoding='utf-8') as fp:
            # skip the head line
            next(fp)
            for line in fp.readlines():
                words, labels, event_types = line.strip('\n').split('\t')
                words = words.split('\002')
                labels = labels.split('\002')
                event_types = event_types.split('\002')
                self.word_ids.append(words)
                self.label_ids.append(labels)
                self.event_types.append(event_types)

    def __len__(self):
        return len(self.word_ids)

    def __getitem__(self, index):
        return self.word_ids[index], self.label_ids[index], self.event_types[index]




def data_process(path, model="trigger", is_predict=False):
    """data_process"""

    def label_data(data, start, length, _type):
        """label_data"""
        for i in range(start, start + length):
            suffix = "B-" if i == start else "I-"
            data[i] = "{}{}".format(suffix, _type)
        return data

    def label_border(data, start, length):
        """label trigger word's borders"""
        data[start] = 'B-边界'
        data[start+length-1] = 'E-边界'
        for i in range(start+1, start+length-1):
            data[i] = 'I-边界'
        return data



    sentences = []
    output = ["text_a"] if is_predict else ["text_a\tlabel"]
    with open(path, encoding='utf-8') as f:
        for line in f:
            d_json = json.loads(line.strip())
            _id = d_json["id"]
            text_a = [
                "，" if t == " " or t == "\n" or t == "\t" else t
                for t in list(d_json["text"].lower())
            ]
            if is_predict:
                sentences.append({"text": d_json["text"], "id": _id})
                output.append('\002'.join(text_a))
            else:
                if model == "trigger":
                    labels = ["O"] * len(text_a)
                    #border_labels = ["O"] * len(text_a)
                    event_types = []
                    for event in d_json["event_list"]:
                        event_type = event["event_type"]
                        start = event["trigger_start_index"]
                        trigger = event["trigger"]
                        labels = label_data(labels, start,
                                            len(trigger), event_type)
                        # border_labels = label_border(border_labels, start,
                        #                     len(trigger))
                        event_types.append(event_type)
                    output.append("{}\t{}\t{}".format('\002'.join(text_a),
                                                  '\002'.join(labels),
                                                  '\002'.join(event_types)))
                elif model == "role":
                    for event in d_json["event_list"]:
                        labels = ["O"] * len(text_a)
                        for arg in event["arguments"]:
                            role_type = arg["role"]
                            argument = arg["argument"]
                            start = arg["argument_start_index"]
                            labels = label_data(labels, start,
                                                len(argument), role_type)
                        output.append("{}\t{}".format('\002'.join(text_a),
                                                      '\002'.join(labels)))
    return output


def schema_process(path, model="trigger"):
    """schema_process"""

    def label_add(labels, _type):
        """label_add"""
        if "B-{}".format(_type) not in labels:
            labels.extend(["B-{}".format(_type), "I-{}".format(_type)])
        return labels

    labels = []
    for line in read_by_lines(path):
        d_json = json.loads(line.strip())
        if model == "trigger":
            labels = label_add(labels, d_json["event_type"])
        elif model == "role":
            for role in d_json["role_list"]:
                labels = label_add(labels, role["role"])
    labels.append("O")
    tags = []
    for index, label in enumerate(labels):
        tags.append("{}\t{}".format(index, label))
    return tags


def create_event_dict_from_existed_trigger_dict(dict_file):
    new_dict_file = os.path.join(os.path.dirname(dict_file), 'event_tag.dict')
    with open(new_dict_file, mode='w', encoding='utf-8') as writer:
        with open(dict_file, mode='r', encoding='utf-8') as reader:
            for i, line in enumerate(reader):
                if i % 2 == 0:
                    id_, label = line.strip().split('\t')
                    
                    new_label = label[2:]
                    if len(label) == 1:
                        new_label = label
                    writer.write(f"{int(i/2)}\t{new_label}\n")





if __name__ == "__main__":
    print("\n=================DUEE 1.0 DATASET==============")
    
    conf_dir = os.path.join(os.path.dirname(__file__), 'data')
    # 'E:/PythonProjects/BaiduIECompetition/data/event_extraction'
    schema_path = "{}/duee_event_schema.json".format(conf_dir)
    tags_trigger_path = "{}/trigger_bio_tag.dict".format(conf_dir)
    tags_role_path = "{}/role_tag.dict".format(conf_dir)
    print("\n=================start schema process==============")
    print('input path {}'.format(schema_path))
    tags_trigger = schema_process(schema_path, "trigger")
    write_by_lines(tags_trigger_path, tags_trigger)
    print("save trigger tag {} at {}".format(
        len(tags_trigger), tags_trigger_path))
    tags_role = schema_process(schema_path, "role")
    write_by_lines(tags_role_path, tags_role)
    print("save trigger tag {} at {}".format(len(tags_role), tags_role_path))
    print("=================end schema process===============")

    # data process
    data_dir = conf_dir
    trigger_save_dir = "{}/trigger".format(data_dir)
    role_save_dir = "{}/role".format(data_dir)
    print("\n=================start end event annotation process==============")
    if not os.path.exists(trigger_save_dir):
        os.makedirs(trigger_save_dir)

    print("\n----trigger------for dir {} to {}".format(data_dir,
                                                       trigger_save_dir))
    train_tri = data_process("{}/train.json".format(data_dir), "trigger")
    write_by_lines("{}/train.tsv".format(trigger_save_dir), train_tri)
    dev_tri = data_process("{}/dev.json".format(data_dir), "trigger")
    write_by_lines("{}/dev.tsv".format(trigger_save_dir), dev_tri)
    # test_tri = data_process("{}/test.json".format(data_dir), "trigger", is_predict=True)
    # write_by_lines("{}/test.tsv".format(trigger_save_dir), test_tri)
    create_event_dict_from_existed_trigger_dict(tags_trigger_path)
    
    print("train {} dev {}".format(
        len(train_tri), len(dev_tri)))

    # if not os.path.exists(role_save_dir):
    #     os.makedirs(role_save_dir)
    # print("\n----role------for dir {} to {}".format(data_dir, role_save_dir))
    # train_role = data_process("{}/train.json".format(data_dir), "role")
    # write_by_lines("{}/train.tsv".format(role_save_dir), train_role)
    # dev_role = data_process("{}/dev.json".format(data_dir), "role")
    # write_by_lines("{}/dev.tsv".format(role_save_dir), dev_role)
    # test_role = data_process("{}/test.json".format(data_dir), "role", is_predict=True)
    # write_by_lines("{}/test.tsv".format(role_save_dir), test_role)
    # print("train {} dev {} test {}".format(
    #     len(train_role), len(dev_role), len(test_role)))
    # print("=================end event annotation process==============")
