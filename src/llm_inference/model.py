# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from transformers import AutoModel, AutoModelForMaskedLM, AutoModelForQuestionAnswering


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class TaggingModel(nn.Module):

    def __init__(self, config, args):
        super(TaggingModel, self).__init__()
        self.config = config
        self.args = args
        self.loss_weight = args.loss_weight

        self.encoder = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path, config=config)  ## (batch, 512, 51257)
        # self.encoder = AutoModel.from_pretrained(args.model_name_or_path, config=config)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, source_ids, target_ids):
        encoder_output = self.encoder(source_ids,
                                      attention_mask=source_ids.ne(1)[:,None,:]*source_ids.ne(1)[:,:,None],
                                      labels=target_ids)
        logits = encoder_output.logits
        predicted_labels = torch.argmax(logits, dim=-1)

        loss1 = self.loss_fct(logits.view(-1, self.config.vocab_size), target_ids.view(-1))  # =encoder_output.loss
        mask = source_ids != target_ids
        loss2 = self.loss_fct(logits.view(-1, self.config.vocab_size)[mask.view(-1)], target_ids.view(-1)[mask.view(-1)])
        return loss1 + loss2 * self.loss_weight, predicted_labels


def update_special_tokens(model, tokenizer, special_tokens=["i-desc", "i-param"], initialize='random', labels=None):
    """Add special tokens to the RoBERTa tokenizer and model"""
    # Update tokenizer with new special tokens
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

    # Update model with new embeddings
    old_num_tokens = model.config.vocab_size
    model.resize_token_embeddings(old_num_tokens + len(special_tokens))

    new_embeddings = model.roberta.embeddings.word_embeddings.weight[old_num_tokens:]
    # new_embeddings = model.embeddings.word_embeddings.weight[old_num_tokens:]
    if initialize == 'random':
        nn.init.normal_(new_embeddings, mean=0, std=model.config.initializer_range)
    elif initialize == 'xavier':
        nn.init.xavier_uniform_(new_embeddings)
    elif initialize == 'labels':
        if labels is None:
            raise Exception("Did not assign 'labels' list to initialize embedding")
        for i, token in enumerate(special_tokens):
            if token in labels:
                token_list = labels[token]
                if token_list is None or len(token_list) == 0:
                    # If label is empty, initialize with random values
                    nn.init.normal_(new_embeddings[i], mean=0, std=model.config.initializer_range)
                else:
                    # If label is not empty, average the embeddings of the tokens and initialize with the average
                    token_indices = [tokenizer.convert_tokens_to_ids(t) if isinstance(t, str) else t for t in token_list]
                    embeddings_sum = 0
                    for index in token_indices:
                        embeddings_sum += model.roberta.embeddings.word_embeddings.weight.data[index]
                    embeddings_avg = embeddings_sum / len(token_indices)
                    new_embeddings[i] = embeddings_avg
            else:
                # Initialize new embeddings with random values
                nn.init.normal_(new_embeddings[i], mean=0, std=model.config.initializer_range)

    return model, tokenizer

