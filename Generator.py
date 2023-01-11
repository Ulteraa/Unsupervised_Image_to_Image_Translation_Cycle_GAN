import torch
import torch.nn as nn
class ConvBlock(nn.Module):
    def __init__(self, in_chenel, out_chenel, down=True, stride=2):
        super(ConvBlock, self).__init__()
        if down:
             self.conv=nn.Sequential(nn.Conv2d(in_chenel,out_chenel,kernel_size=3,stride=stride,padding=1, padding_mode='reflect'),
                                nn.InstanceNorm2d(out_chenel),
                                nn.ReLU())
        else:
            self.conv=nn.Sequential(nn.ConvTranspose2d(in_chenel,out_chenel,3,stride=stride,padding=1,output_padding=1),
                                    nn.InstanceNorm2d(out_chenel),
                                    nn.ReLU())
    def forward(self,x):
        return self.conv(x)
class ResidualBlock(nn.Module):
    def __init__(self, in_chanel):
        super(ResidualBlock, self).__init__()
        self.conv=nn.Sequential(ConvBlock(in_chanel,in_chanel,stride=1),
                                ConvBlock(in_chanel,in_chanel,stride=1))
    def forward(self,x):
        x=x+self.conv(x)
        return x

# c7s1-64,d128,d256,R256,R256,R256,
# R256,R256,R256,R256,R256,R256,u128
# u64,c7s1-3
class Generator(nn.Module):
    def __init__(self,in_chanel, features=[128,256], num_res=9):
        super(Generator, self,).__init__()
        self.initial=nn.Sequential(nn.Conv2d(in_chanel,64,7,1,padding=3,padding_mode='reflect'),
                                   nn.InstanceNorm2d(64),
                                   nn.ReLU())
        layer=[]
        in_chanel=64
        for feature in features:
            layer.append(ConvBlock(in_chanel,feature,stride=2))
            in_chanel=feature
        for i in range(num_res):
            layer.append(ResidualBlock(in_chanel))

        self.conv_done=nn.Sequential(*layer)

        self.conv_up=nn.Sequential(ConvBlock(in_chanel,128,down=False),
                                   ConvBlock(128,64,down=False))

        self.last=nn.Conv2d(64,3,7,1,padding=3,padding_mode='reflect')
    def forward(self,x):
        x=self.initial(x)
        x=self.conv_done(x)
        x=self.conv_up(x)
        x=torch.tanh(self.last(x))
        return x

# def test():
#     x=torch.randn((1,3,256,256))
#     in_chanel=3
#     # features=[64,128,256,512]
#     model=Generator(in_chanel)
#     predict=model(x)
#     print(predict.shape)
# if __name__=='__main__':
#     test()



