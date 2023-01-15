import  torch
import  torch.nn as nn
# C64-C128-C256-C512
class ConvBlock(nn.Module):
    def __init__(self,in_chanel,out_chanel,strid=2):
        super(ConvBlock, self).__init__()
        self.conv=nn.Sequential(nn.Conv2d(in_chanel,out_chanel,4,stride=strid,padding=1,padding_mode='reflect'),
                                nn.InstanceNorm2d(out_chanel),
                                nn.LeakyReLU(0.2))
    def forward(self,x):
        return self.conv(x)
class Discriminator(nn.Module):
    def __init__(self,in_chanel,features=[64,128,256,512]):
        super(Discriminator, self).__init__()
        self.initial=nn.Sequential(nn.Conv2d(in_chanel,features[0],4,2,1, padding_mode='reflect'),
                                   nn.LeakyReLU(0.2))
        layer=[];in_chanel=features[0]
        for feature in features[1:]:
            layer.append(ConvBlock(in_chanel,feature,strid=1 if feature==features[-1] else 2))
            in_chanel=feature
        self.conv=nn.Sequential(*layer)
        self.final=nn.Conv2d(in_chanel,1,4,1,1, padding_mode='reflect')

    def forward(self,x):
        x=self.initial(x)
        x=self.conv(x)
        x=self.final(x)
        return torch.sigmoid(x)

# def test():
#     x=torch.randn((15,3,256,256))
#     in_chanel=3
#     # features=[64,128,256,512]
#     model=Discriminator(in_chanel)
#     predict=model(x)
#     print(predict.shape)
# if __name__=='__main__':
#     test()
