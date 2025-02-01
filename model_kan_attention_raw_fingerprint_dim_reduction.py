import torch
from dgllife.model import GAT
from dgl.nn.pytorch import MaxPooling
import torch.nn as nn
import kan as kan
from self_attention_2 import SelfAttention

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def squash2(x,dim=-1):
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    u = (1-1/torch.exp(squared_norm.sqrt()))*x / (squared_norm+1e-8).sqrt()
    return u

class GAT1(nn.Module):

    def __init__(self, n_feats, fp, gfp):
        super(GAT1,self).__init__()
        self.n_feats = n_feats
        self.fp = fp
        self.gfp = gfp
        self.gnn_layers = GAT(
            in_feats=self.n_feats,
            hidden_feats=[50,50],
            num_heads=[4,4],
            feat_drops=[0.2,0.2],
            attn_drops=[0.2,0.2],
            alphas=[0.2,0.2],
            residuals=[True,True],
            agg_modes=["flatten", "mean"],
            activations=[nn.LeakyReLU(), None]
        )
        
        self.pool = MaxPooling()
        self.linear = nn.Sequential(
          #nn.Linear(50 + self.fp, 128),
          nn.Linear(50 + 256, 128),
                                    nn.Dropout(0.2),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(128),
                                    nn.Linear(128,128),
                                    nn.ReLU())
        self.fpsMLP = nn.Sequential(nn.Linear(fp, 256), nn.ReLU())

    def forward(self, input_d, fps):
        graph = input_d
        out = graph.ndata["h"]
        out = self.gnn_layers(graph, out)
        out = self.pool(graph,out)
        fpsout = self.fpsMLP(fps)
        out = torch.cat([out, fpsout], dim=-1)
        out = self.linear(out)
        return out

class Mymodel(nn.Module):

    def __init__(self, n_feats, fp, gfp):
        super(Mymodel, self).__init__()
        self.GAT = GAT1(n_feats, fp, gfp)
        # self.pri = PrimaryCaps2(out_channels=8)
        # self.dig = DigitCaps2(in_dim=8,
        #                      in_caps=16,
        #                      num_caps=2,
        #                      dim_caps=2,
        #                      D = 128)
        self.fuse_no_extension = kan.KANLinear(384, 1)
        self.fp_attention = SelfAttention(939, 512, 939)
        #self.fp_attention = SelfAttention(1126, 512, 1126)
        
        #nn.Sequential(
        #   nn.Linear(384, 128),
        #   nn.ReLU(),
        #   nn.Linear(128, 2)
        #)
        self.gfpMLP = kan.KANLinear(gfp, 256)
        #nn.Sequential(nn.Linear(gfp, 1024), nn.ReLU(), nn.Linear(1024, 256), nn.ReLU())

    def forward(self, x, fp, gfp):
        #breakpoint()
        gfp = torch.squeeze(gfp, dim=1)
        #breakpoint()
        fp = fp.unsqueeze(dim=1)
       
        fpweight = self.fp_attention(fp)
        #breakpoint()
        fp = fp * fpweight
        fp = fp.squeeze(dim=1)

        out = self.GAT(x,fp)
        gfpout = self.gfpMLP(gfp)
        out = torch.concat([out, gfpout], axis=1)
        fout = self.fuse_no_extension(out)
        return fout
        #out = self.fuse_no_extension(out)
        # out = self.pri(out)
        #breakpoint()
        # out = self.dig(out)
        #breakpoint()

        # logits = (out ** 2).sum(dim=-1)
        # logits = (logits + 1e-8).sqrt()

        # return logits