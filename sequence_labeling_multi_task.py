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
"""
sequence labeling
"""
import argparse
import ast
import os
import shutil
import warnings
from collections import defaultdict
from functools import partial
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer

from data_prepare import (DuEventExtraction,
                          convert_example_to_feature_multitask_event_type)
from losses import FocalLoss
from models.bert import BertForDD
from models.multi_task_bert import (MultiTaskBertCrfForDD,
                                    MultiTaskBertCrfWithConstraintForDD,
                                    MultiTaskBertForDD)
from sequence_labeling import (HyperparasForModelSearch, ModelTestOutput,
                               ModelTrainOutput)
from utils import (ChunkEvaluator, EarlyStopping, Pad, Stack, TrainingRecord,
                   Tuple, create_folders_if_not_exist, load_dict,
                   seed_everywhere)

warnings.filterwarnings('ignore')

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch", type=int, default=5, help="Number of epoches for fine-tuning.")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate used to train with warmup.")
parser.add_argument("--tag_path", type=str, default=None, help="tag set path")
parser.add_argument("--vocab_path", type=str, default=None, help="vocab path")
parser.add_argument("--model_name", type=str, default='bert', help="model name")
parser.add_argument("--data_path", type=str, default='./data', help="train data")
parser.add_argument("--do_train", type=ast.literal_eval, default=True, help="do train")
parser.add_argument("--do_predict", type=ast.literal_eval, default=False, help="do predict")
parser.add_argument("--do_badcase_analysis", type=ast.literal_eval, default=False, help="do badcase analysis")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay rate for L2 regularizer.")
parser.add_argument("--warmup_proportion", type=float, default=0.1, help="Warmup proportion params for warmup strategy")
parser.add_argument("--max_seq_len", type=int, default=256, help="Number of words of the longest seqence.")
parser.add_argument("--valid_step", type=int, default=100, help="validation step")
parser.add_argument("--skip_step", type=int, default=50, help="skip step")
parser.add_argument("--train_batch_size", type=int, default=8, help="Total examples' number in batch for training.")
parser.add_argument("--eval_batch_size", type=int, default=16, help="Total examples' number in batch for validating.")
parser.add_argument("--init_ckpt", type=str, default=None, help="already pretraining model checkpoint")
parser.add_argument("--output_dir", default="outputs", type=str, required=False, help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--predict_save_path", type=str, default=None, help="predict data save path")
parser.add_argument("--seed", type=int, default=24, help="random seed for initialization")
parser.add_argument("--n_gpu", type=int, default=1, help="Number of GPUs to use, 0 for CPU.")
args = parser.parse_args()
# yapf: enable.



def do_train(model, train_loader, dev_loader, criterion, multi_label_criterion, optimizer, metric, scheduler):
    '''模型训练
    '''
    logger.info("============start train==========")
    num_training_steps = len(train_loader) * args.num_epoch

    model_save_path = os.path.join(args.checkpoint_dir, "pytorch_model.bin")
    early_stopping = EarlyStopping(patience=10, metric='f1', trace_func=logger.info, path=model_save_path)
    training_record = TrainingRecord(os.path.join(args.record_dir, f'{args.model_name}_{args.pretrained_model}_training'), logger.info)
    train_output = ModelTrainOutput()
    step, best_f1 = 0, 0.0
    model.to(args.device).train()
    for epoch in range(args.num_epoch):
        for idx, (input_ids, token_type_ids, seq_lens, labels, event_labels) in enumerate(train_loader):
            input_ids, labels, event_labels, token_type_ids = (
                torch.LongTensor(input_ids).to(args.device), 
                torch.LongTensor(labels).to(args.device),
                torch.FloatTensor(event_labels).to(args.device),
                torch.LongTensor(token_type_ids).to(args.device)
            )
            
            masks = [[1 if i < length else 0 for i in range(input_ids.size(1))] for length in seq_lens]
            masks = torch.FloatTensor(masks).to(args.device)
            optimizer.zero_grad()
            result = model(input_ids, masks, token_type_ids=token_type_ids, labels=labels)
            if result.loss is not None:
                loss = result.loss
            else:
                logits = result.logits.view([-1, result.logits.size(-1)])
                loss = torch.mean(criterion(logits, labels.view([-1])))
            if result.event_multi_logits is not None:
                
                event_loss = multi_label_criterion(result.event_multi_logits, event_labels)
                loss = args.epsilon * loss + (1-args.epsilon) * event_loss
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            
            loss_item = loss.item()
            if step > 0 and step % args.skip_step == 0:
                logger.info(f'train epoch: {epoch+1} - step: {step} (total: {num_training_steps}) - loss: {loss_item:.6f}')
            if step > 0 and step % args.valid_step == 0:
                p, r, f1, loss_avg = evaluate(model, criterion, multi_label_criterion, metric, dev_loader)
                
                # adjust the learning rate
                scheduler.step(loss_avg)
                # 记录训练时的指标输出
                if f1 > best_f1:
                    best_f1 = f1
                    train_output.dev_recall, train_output.dev_prec, train_output.dev_f1, train_output.epochs = (
                        r, p, f1, epoch+1
                    )
                training_record.add_record(**{'loss': loss_avg, 'precision': p, 'recall': r, 'f1': f1})
                
                logger.info(f'dev step: {step} - loss: {loss_avg:.5f}, precision: {p:.5f}, recall: {r:.5f}, ' \
                        f'f1: {f1:.5f}， current best f1: {best_f1:.5f}')
                
                # early_stopping 会负责保存最优模型
                if early_stopping(f1, model):
                    logger.info("Stopped training due to the loss didn't decrease for a long time.")
                    break
            step += 1

        if early_stopping.stop():
            logger.info('Stopped training.')
            break   # 若是触发 early_stopping 则会执行这里的 break 以跳出最外层 for 训练结束训练

    training_record.save_as_imgs()
    logger.info("=====training complete=====")
    return train_output

@torch.no_grad()
def evaluate(model, criterion, multi_label_criterion, metric, data_loader):
    """模型评估
    """
    model.eval()
    metric.reset()
    losses = []
    for idx, (input_ids, token_type_ids, seq_lens, labels, event_labels) in enumerate(data_loader):
        input_ids, labels, event_labels, token_type_ids = (
            torch.LongTensor(input_ids).to(args.device), 
            torch.LongTensor(labels).to(args.device),
            torch.FloatTensor(event_labels).to(args.device),
            torch.LongTensor(token_type_ids).to(args.device)
        )
        masks = [[1 if i < length else 0 for i in range(input_ids.size(1))] for length in seq_lens]
        masks = torch.FloatTensor(masks).to(args.device)
        result = model(input_ids, masks, token_type_ids=token_type_ids, labels=labels, is_predict=False)
        if result.loss is not None:
            loss = result.loss
        else:
            logits = result.logits.view([-1, result.logits.size(-1)])
            loss = torch.mean(criterion(logits, labels.view([-1])))

        if result.event_multi_logits is not None:
                
            event_loss = multi_label_criterion(result.event_multi_logits, event_labels)
            loss = args.epsilon * loss + (1-args.epsilon) * event_loss
        losses.append(loss.item())
        if result.predictions is not None:
            preds = result.predictions
        else:
            preds = torch.argmax(result.logits, axis=-1)
        n_infer, n_label, n_correct = metric.compute(None, seq_lens, preds.cpu(), labels.cpu())
        metric.update(n_infer, n_label, n_correct)
        precision, recall, f1_score = metric.accumulate()
    avg_loss = np.mean(losses)
    model.train()

    return precision, recall, f1_score, avg_loss


@torch.no_grad()
def badcase_analysis(model, save_path, data_loader, metric):
    model.to(args.device).eval()
    metric.reset()
    writer = open(f'{save_path}/{args.model_name}_{args.pretrained_model}_badcase.txt', mode='w', encoding='utf-8')
    bound_false_total, false_total = 0, 0
    right_for_event_num, total_for_event_num = defaultdict(int), defaultdict(int)
    for idx, (input_ids, token_type_ids, seq_lens, labels, border_labels) in enumerate(data_loader):
        input_ids, labels, border_labels, token_type_ids = (
            torch.LongTensor(input_ids).to(args.device), 
            torch.LongTensor(labels).to(args.device),
            torch.LongTensor(border_labels).to(args.device),
            torch.LongTensor(token_type_ids).to(args.device))
        masks = [[1 if i < length else 0 for i in range(input_ids.size(1))] for length in seq_lens]
        masks = torch.FloatTensor(masks).to(args.device)
        result = model(input_ids, masks, token_type_ids=token_type_ids, labels=labels, is_predict=True)

        if result.predictions is not None:
            preds = result.predictions
        else:
            preds = torch.argmax(result.logits, axis=-1)
        
        n_infer, n_label, n_correct = metric.compute(None, seq_lens, preds.cpu(), labels.cpu())
        n_false = (n_infer - n_correct).item()
        metric.update(n_infer, n_label, n_correct)
        precision, recall, f1_score = metric.accumulate()
        check_result = metric.check(None, seq_lens, preds.cpu(), labels.cpu())
        event_num = n_label.item()
        if event_num >= 4:
            pass
        if check_result:
            
            unpad_predictions, unpad_labels, n_bound_false = check_result
            
            writer.write(f"{idx+1}:\t边界错 {n_bound_false} 个，类别错 {n_false - n_bound_false} 个\n\n")
            writer.write(f"Gold:\t{unpad_labels[0]}\n")
            writer.write(f"Pred:\t{unpad_predictions[0]}\n\n")
            writer.flush()
            bound_false_total += n_bound_false
            false_total += n_false
        else:
            right_for_event_num[event_num] += 1
            assert n_false == 0
        total_for_event_num[event_num] += 1


    ratio_dict = {}
    for key in total_for_event_num.keys():
        ratio_dict[key] = right_for_event_num[key] / total_for_event_num[key]
    writer.close()
    logger.warning(f"{'*'*50}")
    logger.warning(f"按事件数正确：{dict(right_for_event_num)}，总数：{dict(total_for_event_num)}，占比：{ratio_dict}")
    logger.warning(f"共预测 {metric.num_infer_chunks} 个，真实有 {metric.num_label_chunks} 个， 预测正确的有 {metric.num_correct_chunks} 个")
    class_false_total = false_total - bound_false_total
    description = f"触发词共错 {false_total} 个，其中分类错 {class_false_total} 个，占比 {class_false_total/false_total:.5f}"
    logger.warning(description)
    logger.warning(f"precision: {precision:.5f}, recall: {recall:.5f}, f1: {f1_score:.5f}")
    logger.warning(f"{'*'*50}")

    return ModelTestOutput(
        precision=precision,
        recall=recall,
        f1=f1_score,
        description=description
    )



def main(model_map):
    
    pretrained_model_path = f"./pretrained_models/{args.pretrained_model}/"
    task = 'trigger' # trigger
    args.tag_path = os.path.join(args.data_path, f'{task}_bio_tag.dict')
    args.event_tag_path = os.path.join(args.data_path, 'event_tag.dict')
    args.label_map = load_dict(args.tag_path)
    args.event_map = load_dict(args.event_tag_path)
    
    # 创建不存在的文件夹
    checkpoint_root = os.path.join(args.output_dir, 'checkpoints')
    model_root = os.path.join(checkpoint_root, args.model_name)
    args.checkpoint_dir = os.path.join(model_root, args.pretrained_model)
    args.record_dir = os.path.join(args.output_dir, 'record_as_imgs')
    log_save_path = os.path.join(args.output_dir, 'logs')
    badcase_save_path = os.path.join(args.output_dir, 'badcases')
    
    create_folders_if_not_exist(
        args.output_dir,
        checkpoint_root,
        model_root,
        args.checkpoint_dir, 
        args.record_dir, 
        log_save_path, 
        badcase_save_path
    )
    model_config_path = os.path.join(args.checkpoint_dir, 'config.json')
    vocab_path = os.path.join(args.checkpoint_dir, 'vocab.txt')
    try:
        if not os.path.exists(model_config_path):
            shutil.copyfile(os.path.join(pretrained_model_path, 'config.json'), model_config_path)
        if not os.path.exists(vocab_path):
            shutil.copyfile(os.path.join(pretrained_model_path, 'vocab.txt'), vocab_path)
    except OSError as oe:
        logger.warning(f'config.json or vocab.txt file are not exist in {pretrained_model_path}')
    
    handler_id = logger.add(
        f"{log_save_path}/{args.model_name}_{args.pretrained_model}_training.log", 
        level="INFO", rotation="10 MB", encoding='utf-8', mode='a'
    )


    id2label = {val: key for key, val in args.label_map.items()}
    num_labels = len(id2label)
    
    model_path = pretrained_model_path
    if args.do_badcase_analysis:
        model_path = args.checkpoint_dir
        args.eval_batch_size = 1
    model = model_map[args.model_name].from_pretrained(
                model_path, **{'num_labels':num_labels, 'max_seq_len': args.max_seq_len, 'event_nums': len(args.event_map)})
    
    tokenizer = BertTokenizer.from_pretrained(model_path)
    
    if args.do_train:
        train_path = os.path.join(args.data_path, task, 'train.tsv')
        dev_path = os.path.join(args.data_path, task, 'dev.tsv')
        train_ds, test_ds = DuEventExtraction(train_path), DuEventExtraction(dev_path)
        train_size = int(0.9 * len(train_ds))
        dev_size = len(train_ds) - train_size
        # 指定 generator 固定住 train dataset 和 dev dataset
        train_ds, dev_ds = random_split(train_ds, [train_size, dev_size], generator=torch.Generator().manual_seed(24))
        # test_ds = DuEventExtraction(args.test_data, args.vocab_path, args.tag_path)
        metric = ChunkEvaluator(id2label_dict=id2label, suffix=False)

        loss_function = FocalLoss(gamma=args.gamma, weight=None, ignore_index=args.ignore_label)
        multi_label_loss_func = nn.BCEWithLogitsLoss()
        
        trans_func = partial(
            convert_example_to_feature_multitask_event_type,
            tokenizer=tokenizer,
            label_vocab=args.label_map,
            event_vocab=args.event_map,
            max_seq_len=args.max_seq_len,
            no_entity_label=args.no_entity_label,
            ignore_label=args.ignore_label,
            is_test=False)

        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]), # input ids
            Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]), # token type ids
            Stack(), # sequence lens
            Pad(axis=0, pad_val=args.ignore_label), # labels
            Stack(), # event_types
        ): fn(list(map(trans_func, samples)))

        
        test_loader = DataLoader(
            dataset=test_ds,
            batch_size=args.eval_batch_size,
            collate_fn=batchify_fn
        )
        # analysis
        if args.do_badcase_analysis:
            
            test_output = badcase_analysis(model, badcase_save_path, test_loader, metric)
            logger.remove(handler_id)
            return test_output

        train_loader = DataLoader(
            dataset=train_ds,
            batch_size=args.train_batch_size,
            shuffle=True,
            collate_fn=batchify_fn)
            
        dev_loader = DataLoader(
            dataset=dev_ds,
            batch_size=args.eval_batch_size,
            collate_fn=batchify_fn)
        # Defines optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        # Learning rate schedular: reduce learning rate when a metric has stopped improving
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='min',
            patience=2
        )
        train_output = do_train(model, train_loader, dev_loader, loss_function, multi_label_loss_func, optimizer, metric, scheduler)
        return train_output
    
    
    logger.remove(handler_id) # 移除 logger 之前的 handler

if __name__ == '__main__':
    
    seed_everywhere()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_map = {
        'bert': BertForDD,
        'multi_task_bert': MultiTaskBertForDD,
        'multi_task_bert_crf': MultiTaskBertCrfForDD,
        'multi_task_bert_crf_with_constraint': MultiTaskBertCrfWithConstraintForDD
    }

    
    #args.model_name = 'bert'
    args.pretrained_model = 'bert_wwm_ext'
    args.no_entity_label = "O"
    args.ignore_label = -1 # 'O' 是最后一个标签，损失函数中会忽略的标签
    searched_times = 0
    record_writer = open('model_search_record_for_multi_task.txt', mode='a', encoding='utf-8')
    for model_name in ['bert']:
        for epsilon in [0.8]: #model_map.keys()
            args.model_name = model_name
            args.learning_rate = 5e-5
            args.epsilon = epsilon
            logger.warning(f"{'*'*50}")
            logger.warning(f"{args.model_name} training")
            logger.warning(f"{'*'*50}")

            #args.loss_function_name = 'cross_entropy_loss'
            # for focal loss
            args.gamma = 0
            args.weight = (1,1)
            
            # args.do_badcase_analysis = False
            model_train_output = main(model_map)
            logger.info(f"model train output: {model_train_output}")

            logger.warning(f"{'*'*50}")
            logger.warning(f"{args.model_name} badcase analysis")
            logger.warning(f"{'*'*50}")

            args.do_badcase_analysis = True
            model_test_output = main(model_map)
            logger.info(f"model test output: {model_test_output}")
            
            hyparas_for_model = HyperparasForModelSearch(
                **model_train_output.__dict__,
                **model_test_output.__dict__,
                model_name=args.model_name,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                batch_size=args.train_batch_size,
                gamma=args.gamma,
                weight=args.weight,
                epsilon=args.epsilon
            )
            hyparas = hyparas_for_model.__dict__
            if searched_times == 0:
                record_writer.write('\t'.join(hyparas.keys()) + '\n')

            hypara_values = [str(value) if not isinstance(value, float) else str(round(value, 5)) for value in hyparas.values()]
            record_writer.write('\t'.join(hypara_values) + '\n')
            record_writer.flush()

            searched_times += 1
            
    record_writer.close()
        
    
