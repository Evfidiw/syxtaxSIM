import os
import random
import numpy as np
import torch
def setup_seed(seed=3407):
    os.environ['PYTHONASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
setup_seed()

import pdb
import json
import torch.nn.functional as F
import torch.nn as nn
import argparse
import logging
import re

from dataset import PhonemeBERTDataset, separate_phonemebert_test_set
from collator import DataCollatorWithPaddingMLM
from loss_functions import SupervisedContrastiveLoss, KLWithSoftLabelLoss
from finetune_on_phonemebert import compute_metrics

from transformers import DataCollatorWithPadding
from transformers import BertTokenizer
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers.trainer_callback import EarlyStoppingCallback
from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer


from test import convert_head_to_span, convert_span


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Net(torch.nn.Module):
    def __init__(self, args, num_labels, sdoimask=None):
        super(Net, self).__init__()
        self.args = args
        self.bert = RobertaModel.from_pretrained(args.model_name)

        self.bert.config.type_vocab_size = 2
        self.bert.embeddings.token_type_embeddings = nn.Embedding(2, self.bert.config.hidden_size)
        self.bert.embeddings.token_type_embeddings.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
        self.bert.config.hidden_dropout_prob = args.dropout
        self.mlp = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.clf_head = nn.Linear(self.bert.config.hidden_size, num_labels)

        self.sdoimask = sdoimask
        #self.orimask = orimask


    def forward(self, **inputs):
        #pdb.set_trace()
        label = inputs.pop('labels')
        pseudo_label = inputs.pop('pseudo_label')

        if self.sdoimask is not None:
            bert_output = self.bert(**inputs, span_mask=self.sdoimask)
        else:
            bert_output = self.bert(**inputs)
        last_hidden = bert_output.last_hidden_state[:, 0]

        logits = self.clf_head(self.mlp(last_hidden))

        """ Calculate Loss """
        ce_loss_fn = nn.CrossEntropyLoss(reduction='mean')
        loss = ce_loss_fn(logits, label)

        #kd_loss_fn = KLWithSoftLabelLoss(self.args.pseudo_label_temperature, self.args.pseudo_weight)
        #pseudo_loss = kd_loss_fn(logits, pseudo_label)
        #loss += pseudo_loss

        #contrastive_loss_fn = SupervisedContrastiveLoss(temperature=self.args.contrastive_temperature)
        #loss += self.args.contrastive_weight * contrastive_loss_fn(last_hidden, label, soft_labels=pseudo_label)   # 86.8
        #loss += self.args.contrastive_weight * contrastive_loss_fn(last_hidden, label)

        return loss, logits


class ContrastiveTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        loss, logits = model(**inputs)
        outputs = {'logits': logits}
        return loss if not return_outputs else (loss, outputs)


class UpdatePseudoLabelCallback(TrainerCallback):
    def __init__(self, trainer, warmup=0) -> None:
        super().__init__()
        self._trainer = trainer
        self.warmup = warmup

    def on_epoch_end(self, args, state, control, **kwargs):
        pred = self._trainer.predict(test_dataset=self._trainer.train_dataset)
        print("\ntrain metric: ", pred[2])

        if state.epoch > self.warmup:
            percent = max(5 * state.epoch, 30)
            self._trainer.train_dataset.update_pseudo_label(pred, 100 - percent, verbose=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        #default='./Phoneme-BERT/phomeme-bert-data/downstream-datasets/atis',
        default='./Phoneme-BERT/phomeme-bert-data/downstream-datasets/trec6',
        #default='./Phoneme-BERT/phomeme-bert-data/downstream-datasets/trec50',
        #default='./Phoneme-BERT/phomeme-bert-data/downstream-datasets/sst',
        type=str, help="dataset directory"
    )
    parser.add_argument(
        "--model_name", default='roberta-base', type=str, help="model to finetune"
    )
    #parser.add_argument(
    #    "--model_path", default='./models/RobertaTrecSix', type=str, help="model to finetune"
    #)
    parser.add_argument(
        "--tokenizer_name", default='roberta-base', type=str, help="model to finetune"
    )
    parser.add_argument(
        "--output_dir", default='runs/finetune/atis', type=str, help="dir to save finetuned model"
    )
    parser.add_argument(
        "--log_training_dir", default='log_results/atis', type=str, help="dir to save finetuned model"
    )
    # parser.add_argument(
    #     "--log_dir", default='log_results/atis', type=str, help="dir to save finetuned model"
    # )
    # parser.add_argument(
    #     "--log_name", default='finetune', type=str, help="dir to save finetuned model"
    # )
    parser.add_argument(
        "--seed", default=42, type=int, help="seed"
    )
    parser.add_argument(
        "-n", default=1, type=int, help="num to run & average"  # 2
    )
    parser.add_argument(
        "--max_epoch", default=10, type=int, help="total number of epoch"  # 10
    )
    parser.add_argument(
        "--train_bsize", default=64, type=int, help="training batch size"  # 32
    )
    parser.add_argument(
        "--eval_bsize", default=128, type=int, help="evaluation batch size"  # 128
    )
    parser.add_argument(
        "--patience", default=10, type=int, help="early stopping patience"
    )
    parser.add_argument(
        "--train_golden", action='store_true', help="train on golden transcript"
    )
    parser.add_argument(
        "--eval_golden", action='store_true', help="eval on golden transcript"
    )
    parser.add_argument(
        "--use_phoneme", action='store_true', help="use phoneme + text sequence"
    )
    parser.add_argument(
        "--input_mask_ratio", default=0, type=float, help="mlm ratio when training"
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="model hidden dropout"
    )
    parser.add_argument(
        "--save_predict", action='store_true', help="save prediction & test text"
    )
    parser.add_argument(
        "--use_contrastive", action='store_true', help="supervised contrastive objective"
    )
    parser.add_argument(
        "--contrastive_temperature", default=0.2, type=float, help="contrastive temperature"
    )
    parser.add_argument(
        "--contrastive_weight", default=0.1, type=float, help="contrastive loss weight vs classification"
    )
    parser.add_argument(
        "--use_pseudo", action='store_true', help="train from pseudo label"
    )
    parser.add_argument(
        "--pseudo_label_temperature", default=5, type=float, help="contrastive temperature"
    )
    parser.add_argument(
        "--pseudo_weight", default=10, type=float, help="contrastive loss weight vs classification"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # Dataset
    print('\n---------- start reading dataset ----------')
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    train_dir = os.path.join(args.dataset_dir, 'train')
    eval_dir = os.path.join(args.dataset_dir, 'valid')
    test_dir = os.path.join(args.dataset_dir, 'test')
    # dirty16
    id2label_dict = {
        "atis": ['atis_abbreviation', 'atis_aircraft', 'atis_airfare', 'atis_airline',
                 'atis_flight', 'atis_flight_time', 'atis_ground_service', 'atis_quantity'],
        "trec6": ['ABBR', 'DESC', 'ENTY', 'HUM', 'LOC', 'NUM'],
        "trec50": ['ABBR:abb', 'ABBR:exp', 'ENTY:animal', 'ENTY:body', 'ENTY:color',
                   'ENTY:cremat', 'ENTY:currency', 'ENTY:dismed', 'ENTY:event', 'ENTY:food',
                   'ENTY:instru', 'ENTY:lang', 'ENTY:letter', 'ENTY:other', 'ENTY:plant',
                   'ENTY:product', 'ENTY:religion', 'ENTY:sport', 'ENTY:substance', 'ENTY:symbol',
                   'ENTY:techmeth', 'ENTY:termeq', 'ENTY:veh', 'ENTY:word', 'DESC:def',
                   'DESC:desc', 'DESC:manner', 'DESC:reason', 'HUM:gr', 'HUM:ind',
                   'HUM:title', 'HUM:desc', 'LOC:city', 'LOC:country', 'LOC:mount',
                   'LOC:other', 'LOC:state', 'NUM:code', 'NUM:count', 'NUM:date',
                   'NUM:dist', 'NUM:money', 'NUM:ord', 'NUM:other', 'NUM:period',
                   'NUM:perc', 'NUM:speed', 'NUM:temp', 'NUM:volsize', 'NUM:weight'],
        "sst": ['__label__1', '__label__2', '__label__3', '__label__4', '__label__5']
    }
    if "atis" in args.dataset_dir:
        id2label = 'atis'
    elif "trec6" in args.dataset_dir:
        id2label = 'trec6'
    elif "trec50" in args.dataset_dir:
        id2label = 'trec50'
    else:
        id2label = 'sst'
    train_dataset = PhonemeBERTDataset(
        tokenizer, train_dir, use_golden=args.train_golden,
        use_phoneme=args.use_phoneme,
        id2label=id2label_dict[id2label]
    )
    eval_dataset = PhonemeBERTDataset(
        tokenizer, eval_dir, use_golden=args.eval_golden,
        use_phoneme=args.use_phoneme,
        id2label=id2label_dict[id2label]
    )
    test_dataset = PhonemeBERTDataset(
        tokenizer, test_dir, use_golden=args.eval_golden,
        use_phoneme=args.use_phoneme,
        id2label=id2label_dict[id2label]
    )

    test_datasets = [test_dataset] + separate_phonemebert_test_set(test_dataset)

    data_collator = DataCollatorWithPaddingMLM(
        tokenizer=tokenizer,
        mlm=args.input_mask_ratio > 0,
        mlm_probability=args.input_mask_ratio
    )
    print('---------- done reading dataset ----------\n')

    setup_seed()  # dirty17 & dirty18; run two times to evaluate the seed
    all_preds = []


    sen_heads = []
    #with open("./parser_output_atis/train_en.classification_headlist.txt") as f:
    with open("./parser_output_trec6/train_en.classification_headlist.txt") as f:
    #with open("./parser_output_trec50/train_en.classification_headlist.txt") as f:
    #with open("./parser_output_atis/train_text_origin_headlist.txt") as f:
    #with open("./parser_output_trec6/train_text_origin_headlist.txt") as f:
        for line in f.readlines():
            tmp_seq = []
            line = line.strip('\n')
            line = re.findall(r'[1-9]\d*|0', line)
            for i in range(len(line)):
                tmp = line[i]
                tmp_seq.append(int(tmp))
            sen_heads.append(tmp_seq)
    # print(sen_heads)
    sen_span = convert_head_to_span(sen_heads)
    # assert test_sen_span == sen_span

    sen_tokens = []
    #with open("./Phoneme-BERT/phomeme-bert-data/downstream-datasets/atis/train/en.classification.txt", 'r') as f:
    with open("./Phoneme-BERT/phomeme-bert-data/downstream-datasets/trec6/train/en.classification.txt", 'r') as f:
    #with open("./Phoneme-BERT/phomeme-bert-data/downstream-datasets/trec50/train/en.classification.txt", 'r') as f:
    #with open("./Phoneme-BERT/phomeme-bert-data/downstream-datasets/atis/train/text_original.txt", 'r') as f:
    #with open("./Phoneme-BERT/phomeme-bert-data/downstream-datasets/trec6/train/text_original.txt", 'r') as f:
        for line in f.readlines():
            tmp = line.strip().split(" ")
            sen_tokens = sen_tokens + tmp

    # print("sen_tokens: ", sen_tokens)
    # print("sen_span: ", sen_span)
    print("-------------------------------------Srart convert 1------------------------------------------------------")
    example = {"sen_token": sen_tokens, "sen_span": sen_span}
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased", do_lower_case=True)
    tokens, input_span_mask, record_mask = convert_span(example, tokenizer)
    span_mask = torch.tensor(input_span_mask, dtype=torch.float).to(device)
    print(span_mask)

    #   --------------------------------------------------------------------------------------------
    '''
    ori_heads = []
    #with open("./parser_output_atis/train_text_origin_headlist.txt") as f:
    with open("./parser_output_trec6/train_text_origin_headlist.txt") as f:
    # with open("./parser_output_trec50/train_en.classification_headlist.txt") as f:
        for line in f.readlines():
            tmp_seq = []
            line = line.strip('\n')
            line = re.findall(r'[1-9]\d*|0', line)
            for i in range(len(line)):
                tmp = line[i]
                tmp_seq.append(int(tmp))
            ori_heads.append(tmp_seq)
    # print(sen_heads)
    ori_span = convert_head_to_span(ori_heads)
    # assert test_sen_span == sen_span

    ori_tokens = []
    #with open("./Phoneme-BERT/phomeme-bert-data/downstream-datasets/atis/train/text_original.txt", 'r') as f:
    with open("./Phoneme-BERT/phomeme-bert-data/downstream-datasets/trec6/train/text_original.txt", 'r') as f:
    # with open("./Phoneme-BERT/phomeme-bert-data/downstream-datasets/trec50/train/en.classification.txt", 'r') as f:
        for line in f.readlines():
            tmp = line.strip().split(" ")
            ori_tokens = ori_tokens + tmp

    print("-------------------------------------Srart convert  2-------------------------------------------------------")
    example = {"sen_token": ori_tokens, "sen_span": ori_span}
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased", do_lower_case=True)
    tokens, input_span_mask, record_mask = convert_span(example, tokenizer)
    ori_mask = torch.tensor(input_span_mask, dtype=torch.float).to(device)
    print(ori_mask)
    '''


    for n in range(args.n):
        print(f'---------- start training: {n} ----------')
        #span_mask = torch.ones(768, 768).to(device)
        #ori_mask = torch.ones(768, 768).to(device)

        #model = Net(args, len(train_dataset.id2label), sdoimask=None)
        model = Net(args, len(train_dataset.id2label), span_mask)

        # Train model
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            overwrite_output_dir=True,
            evaluation_strategy="epoch",  # noz/steps/epoch
            logging_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            num_train_epochs=args.max_epoch,
            per_device_train_batch_size=args.train_bsize,
            per_device_eval_batch_size=args.eval_bsize,
            weight_decay=0.01,  # strength of weight decay
            seed=args.seed + n,
        )
        logging.info('- In oasis')
        print(training_args.learning_rate)
        trainer = ContrastiveTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
        )
        if args.use_pseudo:
            trainer.add_callback(UpdatePseudoLabelCallback(trainer))

        print('---------- start train on train set ----------')
        trainer.train()
        # test_data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        # trainer.data_collator = test_data_collator

        print('---------- start predict on test set ----------')
        trainer.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        keys = ['test_accuracy']
        preds = []
        for i, test_dataset in enumerate(test_datasets):
            pred = trainer.predict(test_dataset=test_dataset)
            pred = {k: pred[2][k] for k in keys}
            preds.append(pred)
        all_preds.append(preds)

    print('---------- start print predict results ----------')
    predictions = {}
    for preds in all_preds:
        for i, pred in enumerate(preds):
            for k, v in pred.items():
                key = k + '-{}'.format(i)
                predictions[key] = predictions.get(key, []) + [np.round(v, 4)]

    print("\n{:>30}\t{:>8}\t{:>8}\t{}".format('metric', 'mean', 'std', 'values'))
    for k, v in predictions.items():
        mean = np.round(np.mean(v), 4)
        std = np.round(np.std(v), 4)
        print("{:>30}\t{:>8}\t{:>8}\t{}".format(k, mean, std, v))
        # f.write("{:>30}\t{:>8}\t{:>8}\t{}\n".format(k, mean, std, v))
