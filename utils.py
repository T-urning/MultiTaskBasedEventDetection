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
import os
import hashlib
import random
import warnings
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict



def seed_everywhere(seed=24):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def create_folders_if_not_exist(*floders_path):
    ''' 批量创建不存在的文件夹
    '''
    for floder in floders_path:
        if not os.path.exists(floder):
            print(f"floder {floder} is not exists, create it now.")
            os.mkdir(floder)

class EarlyStopping(object):
    """Early stops the training if validation loss doesn't improve after a given patience.
    the original source of this code is https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py 
    """
    def __init__(self, patience=7, metric='loss', verbose=True, delta=0, path='pytorch_model.bin', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            metric (str): Metric to evaluate model's performance, e.g. 'loss' or 'f1'
                          Default: 'loss'
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.metric = metric
        self.by_loss = metric == 'loss'
        self.best_score = np.Inf if self.by_loss else 0.0
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, score, model):

        score = -score if self.by_loss else score
        if self.best_score is None:
            self.save_checkpoint(score, model)
            self.best_score = score
            return False
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.save_checkpoint(score, model)
            self.best_score = score
            self.counter = 0
            return False

    def stop(self):
        return self.early_stop

    def save_checkpoint(self, score, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            words = 'loss decreased' if self.by_loss else f'{self.metric} increased'
            self.trace_func(f'Validation {words} ({self.best_score:.6f} --> {score:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)


class TrainingRecord(object):
    """ a class to help recording the information generated during model training.
    this information includes loss, precision, recall and f1 score of model.
    """
    def __init__(self, save_path=None, trac_func=print):
        
        self.save_path = save_path if save_path else 'training'
        self.items = ['loss', 'precision', 'recall', 'f1']
        self.records = {item: [] for item in self.items}
        self.trac_func = trac_func
        
    def add_record(self, **record):
        """ record values.
        """
        for k, v in record.items():
            try:
                self.records[k].append(v)
            except KeyError as e:
                raise KeyError(f"you provided a record item that not in {self.items}!!")
        self._check()
    
    def save_as_imgs(self, save_path=None):
        """ save the record as images.
        """
        save_path = save_path if save_path else self.save_path
        if len(self.records['loss']) > 0:
            save_name = self.save_path + '_loss.jpg'
            plt.clf()
            plt.plot(list(range(1, len(self.records['loss'])+1)), self.records['loss'])
            plt.xlabel('optimization steps')
            plt.ylabel('loss')
            plt.savefig(save_name)
            self.trac_func(f"image saved at {save_name}.")
        plt.clf()
        save_name = self.save_path + '_pre_rec_f1.jpg'
        if len(self.records['precision']) > 0:
            epochs = list(range(1, len(self.records['precision'])+1))
            plt.plot(epochs, self.records['precision'], label='precision')
        if len(self.records['recall']) > 0:
            epochs = list(range(1, len(self.records['recall'])+1))
            plt.plot(epochs, self.records['recall'], label='recall')
        if len(self.records['f1']) > 0:
            epochs = list(range(1, len(self.records['f1'])+1))
            plt.plot(epochs, self.records['f1'], label='f1')
        plt.xlabel('optimization steps')
        plt.ylabel('metric_value')
        plt.legend()
        plt.savefig(save_name)
        self.trac_func(f"image saved at {save_name}.")

    
    def _check(self):
        """ check
        """
        length = 0
        for k, v in self.records.items():
            if len(v) == 0: continue
            if length == 0:
                length = len(v)
            else:
                if len(v) != length:
                    raise ValueError(f"the number of item {k} not equal to others!")
            return True

def extract_tp_actual_correct(y_true, y_pred, suffix, *args):
    entities_true = defaultdict(set)
    entities_pred = defaultdict(set)
    for type_name, start, end in get_entities(y_true, suffix):
        entities_true[type_name].add((start, end))
    for type_name, start, end in get_entities(y_pred, suffix):
        entities_pred[type_name].add((start, end))

    target_names = sorted(set(entities_true.keys()) | set(entities_pred.keys()))

    tp_sum = np.array([], dtype=np.int32)
    pred_sum = np.array([], dtype=np.int32)
    true_sum = np.array([], dtype=np.int32)
    for type_name in target_names:
        entities_true_type = entities_true.get(type_name, set())
        entities_pred_type = entities_pred.get(type_name, set())
        tp_sum = np.append(tp_sum, len(entities_true_type & entities_pred_type))
        pred_sum = np.append(pred_sum, len(entities_pred_type))
        true_sum = np.append(true_sum, len(entities_true_type))

    return pred_sum, tp_sum, true_sum

def get_entities(seq, suffix=False):
    """Gets entities from sequence.

    Args:
        seq (list): sequence of labels.

    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).

    Example:
        >>> from seqeval.metrics.sequence_labeling import get_entities
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """

    def _validate_chunk(chunk, suffix):
        if chunk in ['O', 'B', 'I', 'E', 'S']:
            return

        if suffix:
            if not chunk.endswith(('-B', '-I', '-E', '-S')):
                warnings.warn('{} seems not to be NE tag.'.format(chunk))

        else:
            if not chunk.startswith(('B-', 'I-', 'E-', 'S-')):
                warnings.warn('{} seems not to be NE tag.'.format(chunk))

    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]

    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        _validate_chunk(chunk, suffix)

        if suffix:
            tag = chunk[-1]
            type_ = chunk[:-1].rsplit('-', maxsplit=1)[0] or '_'
        else:
            tag = chunk[0]
            type_ = chunk[1:].split('-', maxsplit=1)[-1] or '_'

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks

def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'E':
        chunk_end = True
    if prev_tag == 'S':
        chunk_end = True

    if prev_tag == 'B' and tag == 'B':
        chunk_end = True
    if prev_tag == 'B' and tag == 'S':
        chunk_end = True
    if prev_tag == 'B' and tag == 'O':
        chunk_end = True
    if prev_tag == 'I' and tag == 'B':
        chunk_end = True
    if prev_tag == 'I' and tag == 'S':
        chunk_end = True
    if prev_tag == 'I' and tag == 'O':
        chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end

def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B':
        chunk_start = True
    if tag == 'S':
        chunk_start = True

    if prev_tag == 'E' and tag == 'E':
        chunk_start = True
    if prev_tag == 'E' and tag == 'I':
        chunk_start = True
    if prev_tag == 'S' and tag == 'E':
        chunk_start = True
    if prev_tag == 'S' and tag == 'I':
        chunk_start = True
    if prev_tag == 'O' and tag == 'E':
        chunk_start = True
    if prev_tag == 'O' and tag == 'I':
        chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start

class ChunkEvaluator(object):
    """ChunkEvaluator computes the precision, recall and F1-score for chunk detection.
    It is often used in sequence tagging tasks, such as Named Entity Recognition(NER).

    Args:
        label_list (list): The label list.
        suffix (bool): if set True, the label ends with '-B', '-I', '-E' or '-S', else the label starts with them.
    """
    def __init__(self, id2label_dict, suffix=False):
        
        self.id2label_dict = id2label_dict# dict(enumerate(label_list))
        self.suffix = suffix
        self.num_infer_chunks = 0
        self.num_label_chunks = 0
        self.num_correct_chunks = 0

    def compute(self, inputs, lengths, predictions, labels):
        labels = labels.numpy()
        predictions = predictions.numpy()
        unpad_labels = [[
            self.id2label_dict[index]
            for index in labels[sent_index][:lengths[sent_index]]
        ] for sent_index in range(len(lengths))]
        unpad_predictions = [[
            self.id2label_dict.get(index, "O")
            for index in predictions[sent_index][:lengths[sent_index]]
        ] for sent_index in range(len(lengths))]

        pred_sum, tp_sum, true_sum = extract_tp_actual_correct(
            unpad_labels, unpad_predictions, self.suffix)
        num_correct_chunks = tp_sum.sum(keepdims=True)
        num_infer_chunks = pred_sum.sum(keepdims=True)
        num_label_chunks = true_sum.sum(keepdims=True)

        return num_infer_chunks, num_label_chunks, num_correct_chunks

    def check(self, inputs, lengths, predictions, labels):
        labels = labels.numpy()
        predictions = predictions.numpy()
        unpad_labels = [[
            self.id2label_dict[index]
            for index in labels[sent_index][:lengths[sent_index]]
        ] for sent_index in range(len(lengths))]
        unpad_predictions = [[
            self.id2label_dict.get(index, "O")
            for index in predictions[sent_index][:lengths[sent_index]]
        ] for sent_index in range(len(lengths))]


        entities_true = defaultdict(set)
        entities_pred = defaultdict(set)
        for type_name, start, end in get_entities(unpad_labels, self.suffix):
            entities_true[type_name].add((start, end))
        for type_name, start, end in get_entities(unpad_predictions, self.suffix):
            entities_pred[type_name].add((start, end))

        target_names = sorted(set(entities_true.keys()) | set(entities_pred.keys()))

        tp_sum = np.array([], dtype=np.int32)
        pred_sum = np.array([], dtype=np.int32)
        true_sum = np.array([], dtype=np.int32)
        bound_false_sum = np.array([], dtype=np.int32)
        
        # 遍历每个类别的触发词 
        for type_name in target_names:
            entities_true_type = entities_true.get(type_name, set())
            entities_pred_type = entities_pred.get(type_name, set())
            tp_sum = np.append(tp_sum, len(entities_true_type & entities_pred_type))
            pred_sum = np.append(pred_sum, len(entities_pred_type))
            true_sum = np.append(true_sum, len(entities_true_type))

            bound_false_count = self.check_boundary(entities_true_type, entities_pred_type)
            
            bound_false_sum = np.append(bound_false_sum, bound_false_count)
            
        
        num_correct_chunks = tp_sum.sum(keepdims=True)
        num_infer_chunks = pred_sum.sum(keepdims=True)
        num_label_chunks = true_sum.sum(keepdims=True)

        num_bound_false = bound_false_sum.sum().item()
        

        if num_correct_chunks < num_infer_chunks:
            return unpad_predictions, unpad_labels, num_bound_false

        return None

    def check_boundary(self, entities_true_type, entities_pred_type):
        tp_set = entities_true_type & entities_pred_type # true positive
        fn_set = entities_true_type ^ tp_set # false negtive 
        fp_set = entities_pred_type ^ tp_set # false positvie
        # 若不是边界识别错误，则输出类别分类错误
        bound_false_count = 0
        for t_start, t_end in fn_set:
            for p_start, p_end in fp_set:
                if t_start <= p_start < t_end or p_start < t_start < p_end:
                    bound_false_count += 1
                    #print(f'{(t_start, t_end)} -> {(p_start, p_end)}')
                    
        
        #class_false_count = len(fp_set) - bound_false_count
        #print(f"总共预测了 {len(entities_pred_type)} 个触发词，其中边界错 {bound_false_count} 个，类别错 {class_false_count} 个")

        return bound_false_count

    def _is_number_or_matrix(self, var):
        def _is_number_(var):
            return isinstance(
                var, int) or isinstance(var, np.int64) or isinstance(
                    var, float) or (isinstance(var, np.ndarray) and
                                    var.shape == (1, ))

        return _is_number_(var) or isinstance(var, np.ndarray)

    def update(self, num_infer_chunks, num_label_chunks, num_correct_chunks):
        """
        This function takes (num_infer_chunks, num_label_chunks, num_correct_chunks) as input,
        to accumulate and update the corresponding status of the ChunkEvaluator object. The update method is as follows:

        .. math::
                   \\\\ \\begin{array}{l}{\\text { self. num_infer_chunks }+=\\text { num_infer_chunks }} \\\\ {\\text { self. num_Label_chunks }+=\\text { num_label_chunks }} \\\\ {\\text { self. num_correct_chunks }+=\\text { num_correct_chunks }}\\end{array} \\\\

        Args:
            num_infer_chunks(int|numpy.array): The number of chunks in Inference on the given minibatch.
            num_label_chunks(int|numpy.array): The number of chunks in Label on the given mini-batch.
            num_correct_chunks(int|float|numpy.array): The number of chunks both in Inference and Label on the
                                                  given mini-batch.
        """
        if not self._is_number_or_matrix(num_infer_chunks):
            raise ValueError(
                "The 'num_infer_chunks' must be a number(int) or a numpy ndarray."
            )
        if not self._is_number_or_matrix(num_label_chunks):
            raise ValueError(
                "The 'num_label_chunks' must be a number(int, float) or a numpy ndarray."
            )
        if not self._is_number_or_matrix(num_correct_chunks):
            raise ValueError(
                "The 'num_correct_chunks' must be a number(int, float) or a numpy ndarray."
            )
        self.num_infer_chunks += num_infer_chunks
        self.num_label_chunks += num_label_chunks
        self.num_correct_chunks += num_correct_chunks

    def accumulate(self):
        """
        This function returns the mean precision, recall and f1 score for all accumulated minibatches.

        Returns:
            float: mean precision, recall and f1 score.
        """
        precision = float(
            self.num_correct_chunks /
            self.num_infer_chunks) if self.num_infer_chunks else 0.
        recall = float(self.num_correct_chunks /
                       self.num_label_chunks) if self.num_label_chunks else 0.
        f1_score = float(2 * precision * recall / (
            precision + recall)) if self.num_correct_chunks else 0.
        return precision, recall, f1_score

    def reset(self):
        """
        Reset function empties the evaluation memory for previous mini-batches.
        """
        self.num_infer_chunks = 0
        self.num_label_chunks = 0
        self.num_correct_chunks = 0

    def name(self):
        """
        Return name of metric instance.
        """
        return "precision", "recall", "f1"

    
def cal_md5(str):
    """calculate string md5"""
    str = str.decode("utf-8", "ignore").encode("utf-8", "ignore")
    return hashlib.md5(str).hexdigest()


def read_by_lines(path):
    """read the data by line"""
    result = list()
    with open(path, "r", encoding='utf-8') as infile:
        for line in infile:
            result.append(line.strip())
    return result


def write_by_lines(path, data):
    """write the data"""
    with open(path, "w", encoding='utf-8') as outfile:
        [outfile.write(d + "\n") for d in data]


def text_to_sents(text):
    """text_to_sents"""
    deliniter_symbols = [u"。", u"？", u"！"]
    paragraphs = text.split("\n")
    ret = []
    for para in paragraphs:
        if para == u"":
            continue
        sents = [u""]
        for s in para:
            sents[-1] += s
            if s in deliniter_symbols:
                sents.append(u"")
        if sents[-1] == u"":
            sents = sents[:-1]
        ret.extend(sents)
    return ret


def load_dict(dict_path):
    """load_dict"""
    vocab = {}
    for line in open(dict_path, 'r', encoding='utf-8'):
        value, key = line.strip('\n').split('\t')
        vocab[key] = int(value)
    return vocab


def extract_result(text, labels):
    """extract_result"""
    ret, is_start, cur_type = [], False, None
    if len(text) != len(labels):
        # 韩文回导致label 比 text要长
        labels = labels[:len(text)]
    for i, label in enumerate(labels):
        if label != u"O":
            _type = label[2:]
            if label.startswith(u"B-"):
                is_start = True
                cur_type = _type
                ret.append({"start": i, "text": [text[i]], "type": _type})
            elif _type != cur_type:
                """
                # 如果是没有B-开头的，则不要这部分数据
                cur_type = None
                is_start = False
                """
                cur_type = _type
                is_start = True
                ret.append({"start": i, "text": [text[i]], "type": _type})
            elif is_start:
                ret[-1]["text"].append(text[i])
            else:
                cur_type = None
                is_start = False
        else:
            cur_type = None
            is_start = False
    return ret


class Stack(object):
    """
    Stack the input data samples to construct the batch. The N input samples
    must have the same shape/length and will be stacked to construct a batch.
    Args:
        axis (int, optional): The axis in the result data along which the input
            data are stacked. Default: 0.
        dtype (str|numpy.dtype, optional): The value type of the output. If it
            is set to None, the input data type is used. Default: None.
    Example:
        .. code-block:: python
            from paddle.incubate.hapi.text.data_utils import Stack
            # Stack multiple lists
            a = [1, 2, 3, 4]
            b = [4, 5, 6, 8]
            c = [8, 9, 1, 2]
            Stack()([a, b, c])
            '''
            [[1 2 3 4]
             [4 5 6 8]
             [8 9 1 2]]
             '''
    """

    def __init__(self, axis=0, dtype=None):
        self._axis = axis
        self._dtype = dtype

    def __call__(self, data):
        """
        Batchify the input data by stacking.
        Args:
            data (list(numpy.ndarray)): The input data samples.
        Returns:
            numpy.ndarray: Stacked batch data.
        """
        data = np.stack(
            data,
            axis=self._axis).astype(self._dtype) if self._dtype else np.stack(
                data, axis=self._axis)
        return data


class Pad(object):
    """
    Return a callable that pads and stacks data.
    Args:
        pad_val (float|int, optional): The padding value. Default: 0.
        axis (int, optional): The axis to pad the arrays. The arrays will be
            padded to the largest dimension at axis. For example, 
            assume the input arrays have shape (10, 8, 5), (6, 8, 5), (3, 8, 5)
            and the axis is 0. Each input will be padded into 
            (10, 8, 5) and then stacked to form the final output, which has
            shape（3, 10, 8, 5). Default: 0.
        ret_length (bool|numpy.dtype, optional): If it is bool, indicate whether
            to return the valid length in the output, and the data type of
            returned length is int32 if True. If it is numpy.dtype, indicate the
            data type of returned length. Default: False.
        dtype (numpy.dtype, optional): The value type of the output. If it is
            set to None, the input data type is used. Default: None.
        pad_right (bool, optional): Boolean argument indicating whether the 
            padding direction is right-side. If True, it indicates we pad to the right side, 
            while False indicates we pad to the left side. Default: True.
    Example:
        .. code-block:: python
            from paddle.incubate.hapi.text.data_utils import Pad
            # Inputs are multiple lists
            a = [1, 2, 3, 4]
            b = [4, 5, 6]
            c = [8, 2]
            Pad(pad_val=0)([a, b, c])
            '''
            [[1. 2. 3. 4.]
                [4. 5. 6. 0.]
                [8. 2. 0. 0.]]
            '''
     """

    def __init__(self,
                 pad_val=0,
                 axis=0,
                 ret_length=None,
                 dtype=None,
                 pad_right=True):
        self._pad_val = pad_val
        self._axis = axis
        self._ret_length = ret_length
        self._dtype = dtype
        self._pad_right = pad_right

    def __call__(self, data):
        """
        Batchify the input data by padding The input can be list of numpy.ndarray. 
        The arrays will be padded to the largest dimension at axis and then
        stacked to form the final output.  In addition, the function will output
        the original dimensions at the axis if ret_length is not None.
        Args:
            data (list(numpy.ndarray)|list(list)): List of samples to pad and stack.
        Returns:
            numpy.ndarray|tuple: If `ret_length` is False, it is a numpy.ndarray \
                representing the padded batch data and the shape is (N, …). \
                Otherwise, it is a tuple, except for the padded batch data, the \
                tuple also includes a numpy.ndarray representing all samples' \
                original length shaped `(N,)`. 
        """
        arrs = [np.asarray(ele) for ele in data]
        original_length = [ele.shape[self._axis] for ele in arrs]
        max_size = max(original_length)
        ret_shape = list(arrs[0].shape)
        ret_shape[self._axis] = max_size
        ret_shape = (len(arrs), ) + tuple(ret_shape)
        ret = np.full(
            shape=ret_shape,
            fill_value=self._pad_val,
            dtype=arrs[0].dtype if self._dtype is None else self._dtype)
        for i, arr in enumerate(arrs):
            if arr.shape[self._axis] == max_size:
                ret[i] = arr
            else:
                slices = [slice(None) for _ in range(arr.ndim)]
                if self._pad_right:
                    slices[self._axis] = slice(0, arr.shape[self._axis])
                else:
                    slices[self._axis] = slice(max_size - arr.shape[self._axis],
                                               max_size)

                if slices[self._axis].start != slices[self._axis].stop:
                    slices = [slice(i, i + 1)] + slices
                    ret[tuple(slices)] = arr
        if self._ret_length:
            return ret, np.asarray(
                original_length,
                dtype="int32") if self._ret_length == True else np.asarray(
                    original_length, self._ret_length)
        else:
            return ret


class Tuple(object):
    """
    Wrap multiple batchify functions together. The input functions will be applied
    to the corresponding input fields.
    
    Each sample should be a list or tuple containing multiple fields. The i'th
    batchify function stored in Tuple will be applied on the i'th field. 
    
    For example, when data sample is (nd_data, label), you can wrap two batchify
    functions using `Tuple(DataBatchify, LabelBatchify)` to batchify nd_data and
    label correspondingly.
    Args:
        fn (list|tuple|callable): The batchify functions to wrap.
        *args (tuple of callable): The additional batchify functions to wrap.
    Example:
        .. code-block:: python
            from paddle.incubate.hapi.text.data_utils import Tuple, Pad, Stack
            batchify_fn = Tuple(Pad(axis=0, pad_val=0), Stack())
    """

    def __init__(self, fn, *args):
        if isinstance(fn, (list, tuple)):
            assert len(args) == 0, 'Input pattern not understood. The input of Tuple can be ' \
                                   'Tuple(A, B, C) or Tuple([A, B, C]) or Tuple((A, B, C)). ' \
                                   'Received fn=%s, args=%s' % (str(fn), str(args))
            self._fn = fn
        else:
            self._fn = (fn, ) + args
        for i, ele_fn in enumerate(self._fn):
            assert callable(
                ele_fn
            ), 'Batchify functions must be callable! type(fn[%d]) = %s' % (
                i, str(type(ele_fn)))

    def __call__(self, data):
        """
        Batchify data samples by applying each function on the corresponding data
        field, and each data field is produced by stacking the field data of samples.
        Args:
            data (list): The samples to batchfy. Each sample should contain N fields.
        Returns:
            tuple: A tuple composed of results from all including batchifying functions.
        """

        assert len(data[0]) == len(self._fn),\
            'The number of attributes in each data sample should contain' \
            ' {} elements'.format(len(self._fn))
        ret = []
        for i, ele_fn in enumerate(self._fn):
            result = ele_fn([ele[i] for ele in data])
            if isinstance(result, (tuple, list)):
                ret.extend(result)
            else:
                ret.append(result)
        return tuple(ret)


class Dict(object):
    """
    Wrap multiple batchify functions together. The input functions will be applied
    to the corresponding input fields.
    
    Each sample should be a dictionary containing multiple fields. Each
    batchify function with key stored in Dict will be applied on the field which has the same key. 
    
    For example, when data sample is {'tokens': tokens, 'labels': labels), you can wrap two batchify
    functions using `Dict({'tokens': DataBatchify, 'labels': LabelBatchify})` to batchify tokens and
    labels correspondingly.
    Args:
        fn (dict of callable): The batchify functions to wrap.
    Example:
        .. code-block:: python
            from paddle.incubate.hapi.text.data_utils import Dict, Pad, Stack
            batchify_fn = Dict({'tokens': Pad(axis=0, pad_val=0), 'labels': Stack()})
    """

    def __init__(self, fn):
        assert isinstance(fn, (dict)), 'Input pattern not understood. The input of Dict must be a dict with key of input column name and value of collate_fn ' \
                                   'Received fn=%s' % (str(fn))

        self._fn = fn

        for col_name, ele_fn in self._fn.items():
            assert callable(
                ele_fn
            ), 'Batchify functions must be callable! type(fn[%d]) = %s' % (
                col_name, str(type(ele_fn)))

    def __call__(self, data):
        """
        Batchify data samples by applying each function on the corresponding data
        field, and each data field is produced by stacking the field data of samples.
        Args:
            data (list): The samples to batchfy. Each sample should contain N fields.
        Returns:
            tuple: A tuple composed of results from all including batchifying functions.
        """

        ret = []
        for col_name, ele_fn in self._fn.items():
            result = ele_fn([ele[col_name] for ele in data])
            if isinstance(result, (tuple, list)):
                ret.extend(result)
            else:
                ret.append(result)
        return tuple(ret)


if __name__ == "__main__":

    # s = "xxdedewd"
    # print(cal_md5(s.encode("utf-8")))
    badcases_path = os.path.join(os.path.dirname(__file__), 'output', 'badcases')
    record_path = os.path.join(os.path.dirname(__file__), 'output', 'record_as_imgs')
    create_folders_if_not_exist(badcases_path, record_path)
