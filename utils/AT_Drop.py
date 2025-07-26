import math

import torch
import torch.nn.functional as F
from torch import nn as nn

class ADdrop_Loss(object):
    def __init__(self, ignore_index):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.add_nosie = True
        
    def loss(self, outputs, input_ids, target_mask):
        loss = None
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]

        # 将labels中不属于target的部分，设为ignore_index，只计算target部分的loss
        labels = torch.where(target_mask == 1, input_ids, self.ignore_index)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return loss, labels
    
    def kl_loss(self, outputs_origin, adv_outputs, target_mask, target_noise_mask):
        # logits: [batch_size, seq_len, vocab_size]
        logits_origin = outputs_origin.logits
        logits_adv = adv_outputs.logits


        target_mask = target_mask.bool()
        target_noise_mask = target_noise_mask.bool()


        flat_mask = target_mask.view(-1)
        flat_noise_mask = target_noise_mask.view(-1)


        logits_origin_flat = logits_origin.view(-1, logits_origin.size(-1))
        logits_adv_flat = logits_adv.view(-1, logits_adv.size(-1))


        origin_logits_selected = logits_origin_flat[flat_mask]         # [N1, vocab_size]
        adv_logits_selected = logits_adv_flat[flat_noise_mask]         # [N2, vocab_size]


        if origin_logits_selected.size(0) != adv_logits_selected.size(0):
            return 0

        origin_log_probs = torch.nn.functional.log_softmax(origin_logits_selected, dim=-1)
        adv_probs = torch.nn.functional.softmax(adv_logits_selected, dim=-1)


        kl_loss = torch.nn.functional.kl_div(adv_probs, origin_log_probs, reduction='batchmean')
        return kl_loss
    
    def LSVR_loss(self, a):
        # 计算奇异值分解
        _, s, _ = torch.svd(a)
        # 计算奇异值的和
        sum_sigma = torch.sum(s, -1)
        # 获取每个奇异值的指数
        exp_sigma, _ = torch.max(s, -1)
        # 计算损失值
        loss = -torch.log(exp_sigma / sum_sigma)
        # 计算损失值的平均值
        avg_loss = torch.mean(loss)
        return avg_loss
        
        
    def __call__(self, model, inputs, return_outputs = True):
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        target_mask = inputs['target_mask']
        # 取到noise的各种信息
        noise_id = inputs['noise_ids']
        noise_mask = inputs['noise_att_mask']
        target_noise_mask = inputs['noise_mask']

        #计算正常样本的损失
        outputs_origin = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        loss_origin, labels_origin = self.loss(outputs_origin, input_ids, target_mask)

        if self.add_nosie:
            noise = torch.zeros([noise_id.size(0),32,noise_id.size(1),noise_id.size(1)]).to(noise_id.device)
            noise = noise.data.new(noise.size()).normal_(0, 1).to(noise_id.device) * 0.01
            noise.requires_grad_()
            outputs = model(input_ids=noise_id, attention_mask=noise_mask, return_dict=True, noise = noise, noise_initialized = True)
            
            loss, labels = self.loss(outputs, noise_id, target_noise_mask)
            
            # 对抗训练部分
            loss.backward(retain_graph=True)
            delta_grad = noise.grad #对噪声向量计算梯度
            norm = delta_grad.norm()
            model.zero_grad()
            
            if (torch.isnan(norm) or torch.isinf(norm)):  #会出现nan值的处理，这里我觉得很有必要，因为我自己搞的时候就发现很容易出现nan值
                # skim this batch
                return loss_origin, outputs_origin, labels_origin
            # get outputs2
            
            new_noise = noise + (delta_grad / norm) * 2e-5  #更新噪声向量
            adv_outputs = model(input_ids=noise_id, attention_mask=noise_mask, return_dict=True, noise = new_noise, noise_initialized = True)
            adv_loss, labels = self.loss(adv_outputs, noise_id, target_noise_mask)
            loss_origin += adv_loss
            kl_loss = self.kl_loss(outputs_origin, outputs, target_mask, target_noise_mask)
            
        
        return loss_origin, outputs_origin, labels_origin