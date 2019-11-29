p)from simpletransformers.ner.ner_utils import get_examples_from_df
p)import os
p)import tqdm
p)from simpletransformers.ner.ner_utils import convert_example_to_feature
p)import torch
p)from torch.utils.data import TensorDataset

p)def trainNer(data,model):
 examples = get_examples_from_df(data)
 
 args=model.args
 cached_features_file = os.path.join(args["cache_dir"], "cached_{}_{}_{}_{}_{}".format("train", args["model_type"], args["max_seq_length"], model.num_labels,len(examples)))

 features = newconvert_examples_to_features(
                examples,
                model.labels,
                args['max_seq_length'],
                model.tokenizer,
                # XLNet has a CLS token at the end
                cls_token_at_end=bool(args["model_type"] in ["xlnet"]),
                cls_token=model.tokenizer.cls_token,
                cls_token_segment_id=2 if args["model_type"] in ["xlnet"] else 0,
                sep_token=model.tokenizer.sep_token,
                # RoBERTa uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                sep_token_extra=bool(args["model_type"] in ["roberta"]),
                # PAD on the left for XLNet
                pad_on_left=bool(args["model_type"] in ["xlnet"]),
                pad_token=model.tokenizer.convert_tokens_to_ids([model.tokenizer.pad_token])[0],
                pad_token_segment_id=4 if args["model_type"] in ["xlnet"] else 0,
                pad_token_label_id=model.pad_token_label_id,
                process_count=1,
                silent=args['silent']
            )

 all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
 all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
 all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
 all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

 dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
  
 return dataset


p)def newconvert_examples_to_features(
        examples,
        label_list,
        max_seq_length,
        tokenizer,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=-1,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
        process_count=1,
        chunksize=500,
        silent=False
    ):

  label_map = {label: i for i, label in enumerate(label_list)}

  examplesnew = [(
        example, 
        label_map, 
        max_seq_length,
        tokenizer,
        cls_token_at_end,
        cls_token,
        cls_token_segment_id,
        sep_token,
        sep_token_extra,
        pad_on_left,
        pad_token,
        pad_token_segment_id,
        pad_token_label_id,
        sequence_a_segment_id,
        mask_padding_with_zero)
        
            for example in examples]

 

  features = list(map(convert_example_to_feature, examplesnew))

  return features
