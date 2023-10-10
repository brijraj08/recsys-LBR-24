import torch
import torch.nn.functional as F
import torch.nn as nn

def Gelu(x):
    return x*torch.sigmoid(1.702*x)

def BCEloss(item_score,negitem_score):
    pos_loss = -torch.mean(torch.log(torch.sigmoid(item_score)+1e-15))
    neg_loss = -torch.mean(torch.log(1-torh.sigmoid(negitem_score)+1e-15))
    loss = pos_loss + neg_loss
    return loss

class one_transfer(nn.Module):
    '''
    It contains two cnn layers
    '''

    def __init__(self,input_dim,out_dim,kernel=2):
        super(one_transfer, self).__init__()
        self.hidden_dim = input_dim
        self.out_channel = 10
        self.conv1 = nn.Conv2d(1,self.out_channel,(kernel,1),stride=1)

        self.out_channel2 = 5
        self.conv2 = nn.Conv2d(self.out_channel,self.out_channel2,(1,1),stride=1)


        self.fc1 = nn.Linear(input_dim*self.out_channel2,512) # 128
        self.fc2 = nn.Linear(512,out_dim)

        print("kernel:",kernel)
    def forward(self,x):
        x = self.conv1(x)
        #x = x.view(-1,self.hidden_dim*self.out_channel)
        x = Gelu(x)

        x = self.conv2(x)
        x = x.view(-1,self.hidden_dim*self.out_channel2)
        x = Gelu(x)


        x = self.fc1(x)
        x = Gelu(x)
        x = self.fc2(x)
        return x

class ConFusion(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(ConFusion, self).__init__()
        self.user_transfer = one_transfer(in_dim,out_dim,kernel=2)
        self.item_transfer = one_transfer(in_dim,out_dim,kernel=2)
    def forward(self,x_t,x_hat,type):
        x = torch.cat((x_t,x_hat),dim=-1)
        x = x.view(-1,1,2,x_t.shape[-1])
        if type == "user":
            x = self.user_transfer(x)
            x_norm = (x ** 2).sum(dim=-1).sqrt()
            x = x / x_norm.detach().unsqueeze(-1)
        elif type == "item":
            x = self.item_transfer(x)
        else:
            raise TypeError("convtransfer has not this type")
        return x


    def run_MF(self, user_weight_last, user_weight_hat, item_weight_last, item_weight_hat, negitem_weight_last,
               negitem_weight_hat,norm=False):

        user_weight_new = self.forward(user_weight_last, user_weight_hat, 'user')
        item_weight_new = self.forward(item_weight_last, item_weight_hat, 'item')
        negitem_weight_new = self.forward(negitem_weight_last, negitem_weight_hat, 'item')

        item_score = torch.mul(user_weight_new, item_weight_new).sum(dim=-1)
        negitem_score = torch.mul(user_weight_new, negitem_weight_new).sum(dim=-1)
        score = item_score - negitem_score
        if norm:
            user_ = (user_weight_new**2).sum(dim=-1).sqrt()
            score = score/user_
        bpr_loss = -torch.sum(F.logsigmoid(score))
        return bpr_loss

if __name__=="__main__":
    x_t = torch.rand([100,64])
    x_hat = torch.rand_like(x_t)
    net = ConFusion(64,64)
    y = net(x_t,x_hat,"user")
