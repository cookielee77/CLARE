import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
import numpy as np

from transformers import Pipeline

filter_tokens = ['</s>']

class FillMaskPipeline(Pipeline):
    """
    Masked language modeling prediction pipeline using ModelWithLMHead head.
    """
    def __init__(
        self,
        model, 
        tokenizer ,
        device: int = 0,
        topk=5,
    ):
        super().__init__(
            model=model.eval(),
            tokenizer=tokenizer,
            device=device,
            binary_output=True
        )
        self.batch_size = 64
        self.topk = topk
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def __call__(self, texts, tokens=None, mask_info=False):
        # input texts with <mask>
        inputs = self.tokenizer.batch_encode_plus(
            texts, add_special_tokens=True, return_token_type_ids=False,
            return_tensors=self.framework, pad_to_max_length=True,
        )
        with torch.no_grad():
            inputs = self.ensure_tensor_on_device(**inputs)
            try:
                outputs = self.model(**inputs)[0].cpu()
            except:
                outputs = []
                input_ids = inputs['input_ids']
                attention_masks = inputs['attention_mask']
                dataset = TensorDataset(input_ids, attention_masks)
                sampler = SequentialSampler(dataset)
                datasetloader = DataLoader(dataset, sampler=sampler, batch_size=self.batch_size)
                for input_ids, input_mask in datasetloader:
                    output = self.model(
                        input_ids=input_ids, attention_mask=input_mask)[0].cpu()
                    outputs.append(output)
                outputs = torch.cat(outputs, 0)

        results = []
        token_probs = []
        batch_size = outputs.size(0)
        
        if tokens is None:
            tokens = ["" for i in range(batch_size)]

        for i in range(batch_size):
            input_ids = inputs["input_ids"][i]
            result = []
            token_prob = []
            forbid_tokens = [tokens[i]] + filter_tokens

            masked_index = (input_ids == self.tokenizer.mask_token_id).nonzero().item()
            logits = outputs[i, masked_index, :]
            probs = logits.softmax(dim=0)
            values, predictions = probs.topk(self.topk)
            predictions = predictions.tolist()
            
            for j in range(len(predictions)):
                de_token = self.tokenizer.decode(predictions[j]).strip()
                # remove empty and original prediction
                if de_token and de_token not in forbid_tokens:
                    result.append(de_token)
                    token_prob.append(values[j].item())
            results.append(result)
            token_probs.append(np.array(token_prob))
        return results, token_probs
    
    def get_similarity(self, texts, origin_token):
        # remove <s> and </s> symbols
        origin_token_ids = self.tokenizer.encode(origin_token)[1:-1]
        origin_token_length = len(origin_token_ids)
        
        # input texts with synonyms
        inputs = self.tokenizer.batch_encode_plus(
            texts, add_special_tokens=True, return_token_type_ids=False,
            return_tensors=self.framework, pad_to_max_length=True,
        )
        # check the masked index by orginal sentence
        for i in range(len(inputs['input_ids'][0])):
            if inputs['input_ids'][0][i] != inputs['input_ids'][1][i]:
                masked_index = i
                break
        
        with torch.no_grad():
            inputs = self.ensure_tensor_on_device(**inputs)
            sequence_output = self.model(**inputs, get_feature=True)
        
        if origin_token_length == 1:
            orig_feature = sequence_output[0, masked_index, :]
        else:
            # compare with average embedding if original token has multiple subwords
            orig_feature = sequence_output[0, masked_index:masked_index + origin_token_length, :].mean(dim=0)
            
        syn_features = sequence_output[1:, masked_index, :]
        orig_feature = orig_feature.unsqueeze_(0).expand_as(syn_features)

        semantic_sims = self.cos(orig_feature, syn_features).tolist()
        return semantic_sims
    