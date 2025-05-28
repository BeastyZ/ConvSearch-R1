import torch
from torch import nn
from transformers import (
    RobertaConfig,
    RobertaForSequenceClassification, 
    RobertaTokenizer,
    AutoTokenizer, 
    BertModel,
    DPRQuestionEncoderTokenizer, 
    DPRContextEncoderTokenizer,
    DPRContextEncoder, 
    DPRQuestionEncoder
)

# ANCE model
class ANCE(RobertaForSequenceClassification):
    # class Pooler:   # adapt to DPR
    #     def __init__(self, pooler_output):
    #         self.pooler_output = pooler_output

    def __init__(self, config):
        RobertaForSequenceClassification.__init__(self, config)
        self.embeddingHead = nn.Linear(config.hidden_size, 768) # ANCE has
        self.norm = nn.LayerNorm(768)
        self.apply(self._init_weights)
        self.use_mean = False
    
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def query_emb(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask)
        outputs1 = outputs1.last_hidden_state
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        query1 = self.norm(self.embeddingHead(full_emb))
        return query1


    def doc_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)
    

    def masked_mean_or_first(self, emb_all, mask):
        if self.use_mean:
            return self.masked_mean(emb_all, mask)
        else:
            return emb_all[:, 0]
    
    def masked_mean(self, t, mask):
        s = torch.sum(t * mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d
    
    def forward(self, input_ids, attention_mask, wrap_pooler=False):
        return self.query_emb(input_ids, attention_mask)


# TCTColBERT model
class TCTColBERT(nn.Module):
    def __init__(self, model_path) -> None:
        super(TCTColBERT, self).__init__()
        self.model = BertModel.from_pretrained(model_path)

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        if "cur_utt_end_position" in kwargs:
            device = outputs.device
            cur_utt_end_positions = kwargs["cur_utt_end_positions"]
            output_mask = torch.zeros(attention_mask.size()).to(device)
            mask_row = []
            mask_col = []
            for i in range(len(cur_utt_end_positions)):
                mask_row += [i] * (cur_utt_end_positions[i] - 3)
                mask_col += list(range(4, cur_utt_end_positions[i] + 1))
                
            mask_index = (
                    torch.tensor(mask_row).long().to(device),
                    torch.tensor(mask_col).long().to(device)
                )
            values = torch.ones(len(mask_row)).to(device)
            output_mask = output_mask.index_put(mask_index, values)
        else:
            output_mask = attention_mask
            output_mask[:, :4] = 0 # filter the first 4 tokens: [CLS] "[" "Q/D" "]"
            
        # sum / length
        sum_outputs = torch.sum(outputs * output_mask.unsqueeze(-1), dim = -2) 
        real_seq_length = torch.sum(output_mask, dim = 1).view(-1, 1)

        return sum_outputs / real_seq_length


'''
Model-related functions
'''

def load_model(model_type, query_or_doc, model_path):
    assert query_or_doc in ("query", "doc")
    if model_type.lower() == "ance":
        config = RobertaConfig.from_pretrained(
            model_path,
            finetuning_task="MSMarco",
        )
        tokenizer = RobertaTokenizer.from_pretrained(
            model_path,
            do_lower_case=True
        )
        model = ANCE.from_pretrained(model_path, config=config)
    elif model_type.lower() == "dpr-nq":
        if query_or_doc == "query":
            tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_path)
            model = DPRQuestionEncoder.from_pretrained(model_path)
        else:
            tokenizer = DPRContextEncoderTokenizer.from_pretrained(model_path)
            model = DPRContextEncoder.from_pretrained(model_path)
    elif model_type.lower() == "tctcolbert":
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = TCTColBERT(model_path)
    else:
        raise ValueError
    
    # tokenizer.add_tokens(["<CUR_Q>", "<CTX>", "<CTX_R>", "<CTX_Q>"])
    # model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model
