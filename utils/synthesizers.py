import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils.toolkit import DeepInversionHook, TensorPool, ImagePool

class LocalSynthesizer(object):
    def __init__(self, model_list, student, generator, cls_clnt_weight_tensor,
                 nz, num_classes1, num_classes2, img_size, save_dir,
                 transform, normalizer, synthesis_batch_size, sample_batch_size,
                 iterations, warmup, lr_g, lr_z, reset_l0, reset_bn, bn_mmt, args):
        self.model_list = model_list
        self.student = student
        self.generator = generator
        self.cls_clnt_weight_tensor = cls_clnt_weight_tensor
        self.nz = nz
        self.num_classes1 = num_classes1
        self.num_classes2 = num_classes2
        self.img_size = img_size
        self.save_dir = save_dir
        self.synthesis_batch_size = synthesis_batch_size
        self.iterations = iterations
        self.lr_g = lr_g
        self.lr_z = lr_z
        self.args = args

        for m in self.model_list:
            m.eval()
        self.generator.train()
        
        self.all_hooks = []
        for i, model in enumerate(self.model_list):
            hooks = []
            for m in model.modules():
                if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                    hooks.append(DeepInversionHook(m, 0.0))
            self.all_hooks.append(hooks)

        if self.args.get("dataset") == "ciciot23":
            self.data_pool = TensorPool(self.save_dir)
        else:
            self.data_pool = ImagePool(self.save_dir)

        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.lr_g)

    def synthesize(self):
        self.generator.train()
        
        z = torch.randn(self.synthesis_batch_size, self.nz).cuda()
        z.requires_grad = True
        optimizer_z = optim.Adam([z], lr=self.lr_z)

        for it in range(self.iterations):
            self.optimizer_G.zero_grad()
            optimizer_z.zero_grad()

            inputs = self.generator(z)
            
            loss_bns = 0.0
            # Forward pass through all models to trigger hooks
            for i, model in enumerate(self.model_list):
                _ = model(inputs)
                model_bns = sum([h.r_feature for h in self.all_hooks[i]])
                # We can weight the BNS loss by the sum of weights for this client
                client_weight = self.cls_clnt_weight_tensor[i].sum() / self.cls_clnt_weight_tensor[i].shape[0]
                loss_bns += model_bns * client_weight

            loss = loss_bns
            
            loss.backward()
            self.optimizer_G.step()
            optimizer_z.step()
            
        with torch.no_grad():
            self.generator.eval()
            final_z = torch.randn(self.synthesis_batch_size, self.nz).cuda()
            generated_data = self.generator(final_z)
            self.data_pool.add(generated_data)
        
        for hooks in self.all_hooks:
            for h in hooks:
                h.tmp_val = None
