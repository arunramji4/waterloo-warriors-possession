import torch
from torch import nn

class PossessionModel(nn.Module):
    def __init__(self,player_dim=10,scene_dim=10,d_model=64,tokens=76):
        super().__init__()
        self.player=nn.Sequential(nn.Linear(player_dim,32),nn.GELU(),nn.Linear(32,32),nn.GELU())
        self.combine=nn.Linear(32*4+scene_dim,d_model)
        self.position=nn.Parameter(torch.randn(1,tokens,d_model)*.02)
        layer=nn.TransformerEncoderLayer(d_model,4,128,.1,batch_first=True,activation='gelu')
        self.temporal=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.output=nn.Sequential(nn.LayerNorm(d_model),nn.Linear(d_model,3))
    def forward(self,player,mask,team,scene):
        h=self.player(player);pooled=[]
        for t in (0,1):
            m=mask&(team==t);w=m.unsqueeze(-1)
            mean=(h*w).sum(-2)/w.sum(-2).clamp_min(1)
            maximum=h.masked_fill(~w,-1e4).amax(-2)
            maximum=torch.where(m.any(-1,keepdim=True),maximum,torch.zeros_like(maximum))
            pooled.extend([mean,maximum])
        x=self.combine(torch.cat(pooled+[scene],dim=-1))+self.position
        h=self.temporal(x)
        n=h.shape[1]
        centre=h[:,n//2] if n%2 else (h[:,n//2-1]+h[:,n//2])/2
        return self.output(centre)
