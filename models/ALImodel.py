import torch
import torch.nn as nn
import torch.nn.functional as F
import glob as glob
import os
import pickle


class ALIModel(nn.Module):
    def __init__(self, dims=(1, 64, 64), n_hidden=512):
        super(ALIModel, self).__init__()
        # Encoder param
        assert len(dims) == 3
        EncKernel = [5, 4, 4, 4, 4, 1, 1]
        EncStride = [1, 2, 1, 2, 1, 1, 1]
        EncDepth = [32, 64, 128, 256, 512, 512, n_hidden]

        # Generator param
        GenKernel = [4, 4, 4, 4, 5, 1, 1]
        GenStride = [1, 2, 1, 2, 1, 1, 1]
        GenDepth = [256, 128, 64, 32, 32, 32, dims[0]]

        # Discriminator X param
        DxKernel = [5, 4, 4, 4, 4]
        DxStride = [1, 2, 1, 2, 1]
        DxDepth = [32, 64, 128, 256, 512]

        # Discriminator Z param
        DzKernel = [1, 1]
        DzStride = [1, 1]
        DzDepth = [512, 512]

        # Concat Discriminator param
        DxzKernel = [1, 1, 1]
        DxzStride = [1, 1, 1]
        DxzDepth = [1024, 1024, 1]

        if dims[2]==64:
            # Encoder param
            EncKernel = [2, 7, 5, 7, 4, 1]
            EncStride = [1, 2, 2, 2, 1, 1]
            EncDepth = [64, 128, 256, 512, 512, n_hidden]

            # Generator param
            GenKernel = [4, 7, 5, 7, 2, 1]
            GenStride = [1, 2, 2, 2, 1, 1]
            GenDepth = [256, 128, 64, 32, 32, dims[0]]

            # Discriminator X param
            DxKernel = [2, 7, 5, 7, 4]
            DxStride = [1, 2, 2, 2, 1]
            DxDepth = [64, 128, 256, 256, 512]

            # Discriminator Z param
            DzKernel = [1, 1]
            DzStride = [1, 1]
            DzDepth = [512, 512]

            # Concat Discriminator param
            DxzKernel = [1, 1, 1]
            DxzStride = [1, 1, 1]
            DxzDepth = [2048, 2048, 1]

        if dims[2] == 128:
            # Generator param
            GenKernel = [2, 4, 3, 5, 3, 6, 6, 4]
            GenStride = [1, 3, 2, 1, 2, 3, 1, 1]
            GenDepth = [256, 128, 64, 64, 32, 32, 32, dims[0]]

            # Encoder param
            EncKernel = GenKernel[::-1]
            EncStride = GenStride[::-1]
            EncDepth = [64, 128, 128, 256, 256, 512, 512, n_hidden]

            # Discriminator X param
            DxKernel = EncKernel
            DxStride = EncStride
            DxDepth = [64, 128, 128, 256, 256, 512, 512, 512]

            # Discriminator Z param
            DzKernel = [1, 1]
            DzStride = [1, 1]
            DzDepth = [512, 512]

            # Concat Discriminator param
            DxzKernel = [1, 1, 1]
            DxzStride = [1, 1, 1]
            DxzDepth = [2048, 2048, 1]
        self.size = dims[2]
        self.LS = n_hidden
        # Create Model
        self.DisX = DiscriminatorX(KS=DxKernel, ST=DxStride, DP=DxDepth, nc=dims[0])
        self.DisZ = DiscriminatorZ(KS=DzKernel, ST=DzStride, DP=DzDepth, LS=self.LS, nc=dims[0])
        self.DisXZ = DiscriminatorXZ(KS=DxzKernel, ST=DxzStride, DP=DxzDepth, nc=dims[0])
        self.GenZ = Encoder(KS=EncKernel, ST=EncStride, DP=EncDepth, LS=self.LS, nc=dims[0])
        self.GenX = Generator(latent_size=self.LS, KS=GenKernel, ST=GenStride, DP=GenDepth, nc=dims[0])
        self.netid = 'Exp_%d_%d'%(dims[2], self.LS)

    def encode(self, x):
        return self.GenZ(x)

    def generate(self, z):
        return self.GenX(z)

    def sample(self, n=1):
        """
        convenience function for sampling.
        :param n:
        :return:
        """
        z = torch.randn(n, self.LS, 1, 1)
        return self.generate(z)

    def discriminate(self, x):
        z = self.encode(x)
        cat_feature = torch.cat((self.DisZ(z), self.DisX(x)), 1)
        return self.DisXZ(cat_feature)

    def forward(self, x=None, z=None):
        if x is not None and z is None:
            return self.generate(self.encode(x))
        elif x is None and z is not None:
            return self.generate(z)
        elif x is not None and z is not None:
            gen_x = self.generate(z)
            gen_z = self.encode(x)
            p_real = self.DisXZ(torch.cat((self.DisZ(gen_z), self.DisX(x)),1))
            p_fake = self.DisXZ(torch.cat((self.DisZ(z), self.DisX(gen_x)), 1))
            return p_real, p_fake
        else:
            return self.sample(1)


#Generator model (Gx(z))
class Generator(nn.Module):
    def __init__(self, latent_size=32, output_shape=224, nc=1, KS=(4,221), ST =(1,1), DP=(1,1)):
        self.latent_size = latent_size
        self.output_shape = output_shape
        self.nc = nc
        
        super(Generator, self).__init__()
        
        #Build ConvTranspose layer
        self.main = torch.nn.Sequential()
        lastdepth = self.latent_size
        OldDim = 1
        for i in range(len(KS)):
            #Depth
            nnc = DP[i]
            #Kernel Size
            kernel_size = KS[i]
            #Stride
            stride = ST[i]
            
            #Default value
            padding = 0
            output_pading = 0
            
            if i == len(KS)-1:
                nnc = self.nc
                
            #Add ConvTranspose
            self.main.add_module("ConvT_"+str(i), torch.nn.ConvTranspose2d(lastdepth,nnc,kernel_size,stride,padding,output_pading,bias=False))
            
            #Some regurlarisation
            if i != len(KS) - 1:
                self.main.add_module("Relu_"+str(i), torch.nn.ReLU(True))
                self.main.add_module("BN_"+str(i), torch.nn.BatchNorm2d(nnc))            
            #OldDimension (for information)
            OldDim = (OldDim-1)*stride+kernel_size - 2*padding + output_pading
            
            #Last depth (to pass to next ConvT layer)
            lastdepth = nnc
            #print("I=%d K=%d ST=%d Size=%d" % (i,kernel_size,stride,OldDim))
        #self.main.add_module("Sigmoid",nn.Sigmoid())
        self.main.add_module("Tanh",nn.Tanh()) #Apparently Tanh is better than Sigmoid
       
    def forward(self, input):
        return self.main(input)


#Image Encoder network to latent space (Gz(x))
class Encoder(nn.Module):
    def __init__(self,KS,ST,DP,LS, nc):
        super(Encoder, self).__init__()
        
        
        #Sequential model        
        self.main = torch.nn.Sequential()
        lastdepth = nc #This is the number of color (1)
        nnc = 1
        for i in range(len(KS)):
            
            #Kernel, Stride and Depth from param
            kernel_size = KS[i]
            stride = ST[i]
            nnc = DP[i]
            
            #No padding!
            padding = 0
            output_pading = 0
            
            #Conv layer
            self.main.add_module("Conv_"+str(i), 
                                 torch.nn.Conv2d(in_channels=lastdepth,out_channels=nnc,
                                                 kernel_size=kernel_size,stride=stride,bias=False))
           
           #Some regul
            if i != len(KS) - 1:
                self.main.add_module("LRelu_"+str(i), torch.nn.LeakyReLU(0.1, inplace=True))
                self.main.add_module("BN_"+str(i), torch.nn.BatchNorm2d(nnc))
            lastdepth = nnc

    def forward(self, input):
        return self.main(input)


#Discriminator X (Take an image and discriminate it) Dx(x)
class DiscriminatorX(nn.Module):
    def __init__(self,KS,ST,DP, nc):
        super(DiscriminatorX, self).__init__()
        
        self.main = torch.nn.Sequential()
        lastdepth = nc
        nnc = 1
        dp =0.5 #Dropout rate is 0.5 for first
        for i in range(len(KS)):
        
            #Kernel, Stride and Depth from param
            kernel_size = KS[i]
            stride = ST[i]
            nnc = DP[i]
            
            #No padding
            padding = 0
            output_pading = 0
            
            self.main.add_module("Conv_"+str(i), 
                                 torch.nn.Conv2d(in_channels=lastdepth,out_channels=nnc,
                                                 kernel_size=kernel_size,stride=stride,bias=False))
            #Some regularization
            if i != len(KS) - 1:
                self.main.add_module("LRelu_"+str(i), torch.nn.LeakyReLU(0.1, inplace=True))
                self.main.add_module("DropOut_"+str(i), torch.nn.Dropout(dp))
            
            lastdepth = nnc
            dp = 0.2 #New dropout rate
       

    def forward(self, input):
        return self.main(input)

    
#Discriminator for Latent Space (Dz(z))
class DiscriminatorZ(nn.Module):
    def __init__(self,KS,ST,DP,LS, nc):
        super(DiscriminatorZ, self).__init__()
        
        self.main = torch.nn.Sequential()
        lastdepth = LS
        nnc = 1
        dp = 0.5
        for i in range(len(KS)):
            
            #Kernel, Stride and Depth from param
            kernel_size = KS[i]
            stride = ST[i]
            nnc = DP[i]
            
            #No padding!
            padding = 0
            output_pading = 0
            
            #Conv
            self.main.add_module("Conv_"+str(i), 
                                 torch.nn.Conv2d(in_channels=lastdepth,out_channels=nnc,
                                                 kernel_size=kernel_size,stride=stride,bias=False))
           
            if i != len(KS) - 1:
                self.main.add_module("LRelu_"+str(i), torch.nn.LeakyReLU(0.1, inplace=True))
                self.main.add_module("DropOut_"+str(i), torch.nn.Dropout(dp))            
            lastdepth = nnc
            dp = 0.2

    def forward(self, input):
        return self.main(input)


class DiscriminatorXZ(nn.Module):
    def __init__(self,KS,ST,DP, nc):
        super(DiscriminatorXZ, self).__init__()
        
        self.main = torch.nn.Sequential()
        lastdepth = 1024
        nnc = 1
        dp = 0.5
        for i in range(len(KS)):
            
            kernel_size = KS[i]
            stride = ST[i]
            nnc = DP[i]
            
            padding = 0
            output_pading = 0
            
            self.main.add_module("Conv_"+str(i), 
                                 torch.nn.Conv2d(in_channels=lastdepth,out_channels=nnc,
                                                 kernel_size=kernel_size,stride=stride,bias=False))
            
            
            if i != len(KS) - 1:
                self.main.add_module("LRelu_"+str(i), torch.nn.LeakyReLU(0.1, inplace=True))
                self.main.add_module("DropOut_"+str(i), torch.nn.Dropout(dp))

            lastdepth = nnc
            dp = 0.2
        self.main.add_module("Sigmoid", torch.nn.Sigmoid())
           
       

    def forward(self, input):
        return self.main(input)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    load_path = "D:\\Documents\\GitHub\\OD-test-master\\methods\\ALI\\model\\Exp_64_512_0.00001_RandomLabel_4.0\\models"
    model = ALIModel((1, 64, 64), n_hidden=512)
    model.GenX.load_state_dict(torch.load(os.path.join(load_path, "Exp_64_512_0.00001_RandomLabel_4.0_GenX_It_2121000.pth")))
    model.GenZ.load_state_dict(
        torch.load(os.path.join(load_path, "Exp_64_512_0.00001_RandomLabel_4.0_GenZ_It_2121000.pth")))
    model.DisX.load_state_dict(
        torch.load(os.path.join(load_path, "Exp_64_512_0.00001_RandomLabel_4.0_DisX_It_2121000.pth")))
    model.DisZ.load_state_dict(
        torch.load(os.path.join(load_path, "Exp_64_512_0.00001_RandomLabel_4.0_DisZ_It_2121000.pth")))
    model.DisXZ.load_state_dict(
        torch.load(os.path.join(load_path, "Exp_64_512_0.00001_RandomLabel_4.0_DisXZ_It_2121000.pth")))
    #for i in range(10):
    #    sample = model.sample(n=1)
    #    plt.imshow(sample.cpu().squeeze().data.numpy())
    torch.save(model.state_dict(), os.path.join(load_path, "model.best.pth"))



