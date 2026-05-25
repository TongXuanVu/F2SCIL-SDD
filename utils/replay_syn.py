import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils.toolkit import DeepInversionHook, TensorPool, ImagePool

class GlobalSynthesizer(object):
    def __init__(self, teacher, student, generator, nz, num_classes, img_size, init_dataset,
                 save_dir, transform, normalizer, synthesis_batch_size, sample_batch_size,
                 iterations, lr_g, lr_z, reset_l0, reset_bn, bn_mmt, args):
        self.teacher = teacher
        self.student = student
        self.generator = generator
        self.nz = nz
        self.num_classes = num_classes
        self.img_size = img_size
        self.save_dir = save_dir
        self.transform = transform
        self.normalizer = normalizer
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size
        self.iterations = iterations
        self.lr_g = lr_g
        self.lr_z = lr_z
        self.reset_l0 = reset_l0
        self.reset_bn = reset_bn
        self.bn_mmt = bn_mmt
        self.args = args

        self.teacher.eval()
        self.generator.train()
        
        # Attach hooks for BNS
        self.hooks = []
        for m in self.teacher.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                self.hooks.append(DeepInversionHook(m, 0.0))

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
            
            # Forward pass through teacher to trigger hooks
            _ = self.teacher(inputs)

            # Calculate BNS loss
            loss_bns = sum([h.r_feature for h in self.hooks])
            
            # Combine losses
            loss = loss_bns
            
            loss.backward()
            self.optimizer_G.step()
            optimizer_z.step()
            
        # Save generated data
        with torch.no_grad():
            self.generator.eval()
            final_z = torch.randn(self.synthesis_batch_size, self.nz).cuda()
            generated_data = self.generator(final_z)
            self.data_pool.add(generated_data)
        
        # Clear hook temporary values
        for h in self.hooks:
            h.tmp_val = None
