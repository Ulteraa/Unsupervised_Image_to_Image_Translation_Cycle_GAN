import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader
import Generator, Discriminator, dataset_, Config
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
def train():

    m_dataset = dataset_.translation_dataset(Config.root_s,Config.root_t)
    data_loader = DataLoader(dataset=m_dataset,batch_size=Config.batch_size,shuffle=True)
    dis_source = Discriminator.Discriminator(Config.image_chanel)
    dis_target = Discriminator.Discriminator(Config.image_chanel)

    gen_source=Generator.Generator(Config.image_chanel).to(Config.device)
    gen_target= Generator.Generator(Config.image_chanel).to(Config.device)
    l1=nn.L1Loss()
    criterion=nn.MSELoss()
    optim_dis=optim.Adam(list(dis_source.parameters())+list(dis_target.parameters()), lr=Config.lr_,betas=(0.5,0.999))
    optim_gen=optim.Adam(list(gen_source.parameters())+list(gen_target.parameters()), lr=Config.lr_,betas=(0.5,0.999))
    for epoch in range(Config.epochs):
        loop=tqdm(data_loader)
        for _,(source,target) in enumerate(loop):
            source = source.to(Config.device); target = target.to(Config.device)
            target_fake=gen_target(source)
            source_fake=gen_source(target)
            predict_target=dis_target(target)
            predict_target_fake=dis_target(target_fake.detach())

            predict_source = dis_source(source)
            predict_source_fake = dis_source(source_fake.detach())


            # discriminators loss

            loss_dis_source= (criterion(predict_source,torch.ones_like(predict_source))
                              + criterion(predict_source_fake,torch.zeros_like(predict_source_fake)))

            loss_dis_target =(criterion(predict_target,torch.ones_like(predict_target))+
                              criterion(predict_target_fake,torch.zeros_like(predict_target_fake)))

            total_dis_loss=loss_dis_target+loss_dis_source
            optim_dis.zero_grad(total_dis_loss)
            total_dis_loss.backward()
            optim_dis.step()

            # genetors loss

            predict_fake_target=dis_target(target_fake)
            loss_gen_target=criterion(predict_fake_target,torch.ones_like(predict_fake_target))

            predict_fake_source=dis_source(source_fake)

            loss_gen_source= criterion(predict_fake_source, torch.ones_like(predict_fake_source))
            # l1 losse
            l1_loss=l1(source,source_fake)+l1(target,target_fake)

            total_gen_loss=loss_gen_target+loss_gen_source+Config.lambda_*l1_loss

            optim_gen.zero_grad()
            total_gen_loss.backward()
            optim_gen.step()

        print(f'in {epoch}, the dis_loss is {total_dis_loss:4f}, and the gen_loss is {total_gen_loss:4f}')
        with torch.no_grad():
            fake_image_target = gen_target(source)
            fake_image_source = gen_target(target)
            # fake_image_grid = make_grid(fake_image_, normalize=True)
            path_source= 'Cycle_GAN_Results/' + 'image_source' + str(epoch) + '.png'
            save_image(fake_image_source, path_source)

            path_target= 'Cycle_GAN_Results/' + 'image_target' + str(epoch) + '.png'
            save_image(fake_image_target, path_target)


if __name__=='__main__':
    train()


