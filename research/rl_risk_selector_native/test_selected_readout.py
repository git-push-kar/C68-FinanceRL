import unittest
import torch
from torch import nn
from peft.tuners.lora.layer import Linear
from selected_readout import selected_readout

class ReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        self.head=Linear(nn.Linear(7,11,bias=False),'finance',r=3,lora_alpha=6,lora_dropout=0.)
        self.head.base_layer.requires_grad_(False);self.head.eval()
        self.ids=[1,3,7,9];self.h=torch.randn(5,7);self.rows=self.head.base_layer.weight[self.ids].detach().clone()
    def test_zero_delta_preserves_pretrained_logits(self):
        torch.testing.assert_close(selected_readout(self.h,self.head,self.rows,self.ids),nn.functional.linear(self.h,self.rows),atol=0,rtol=0)
    def test_selected_rows_match_full_head_outputs_and_gradients(self):
        with torch.no_grad():self.head.lora_B['finance'].weight.normal_(std=.2)
        full=self.head(self.h)[:,self.ids];full.square().sum().backward()
        grads={n:p.grad.clone() for n,p in self.head.named_parameters() if p.requires_grad}
        self.head.zero_grad();partial=selected_readout(self.h,self.head,self.rows,self.ids);partial.square().sum().backward()
        torch.testing.assert_close(partial,full,atol=1e-6,rtol=1e-6)
        for n,p in self.head.named_parameters():
            if p.requires_grad:torch.testing.assert_close(p.grad,grads[n],atol=1e-5,rtol=1e-5)
    def test_disabling_restores_original_logits(self):
        with torch.no_grad():self.head.lora_B['finance'].weight.normal_(std=.2)
        self.head.enable_adapters(False)
        torch.testing.assert_close(selected_readout(self.h,self.head,self.rows,self.ids),nn.functional.linear(self.h,self.rows),atol=0,rtol=0)

if __name__=='__main__':unittest.main()
