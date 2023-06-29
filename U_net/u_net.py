import torch 
import torch.nn as nn 

def double_convolution(in_channel,out_channel):
    """
    function to implement a sequential double convolution where each convolution is followed by a ReLU activation
    """
    conv = nn.Sequential(nn.Conv2d(in_channel,out_channel, kernel_size=3),
                         nn.ReLU(inplace=True),
                         nn.Conv2d(out_channel,out_channel, kernel_size=3),
                         nn.ReLU(inplace=True))
    return conv    

def crop(img,target_img):
        target_size = target_img.size()[2]
        tensor_size = img.size()[2]
        delta = tensor_size - target_size
        delta = delta // 2
        return img[:,:, delta:tensor_size-delta, delta:tensor_size-delta] 

class Unet(nn.Module):
    """

    """
    def __init__(self):
        super().__init__()
        self.max_pool_2by2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.down_conv_1 = double_convolution(1,64)
        self.down_conv_2 = double_convolution(64,128)
        self.down_conv_3 = double_convolution(128,256)
        self.down_conv_4 = double_convolution(256,512)
        self.down_conv_5 = double_convolution(512,1024)

        self.up_process_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_conv_1 = double_convolution(1024,512)

        self.up_process_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_conv_2 = double_convolution(512,256)

        self.up_process_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_conv_3 = double_convolution(256,128)

        self.up_process_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up_conv_4 = double_convolution(128,64)

        self.output_layer = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)


    def forward(self, image):
        """
        take an image and run the forward pass
        """

        #encoding process
        x1 = self.down_conv_1(image) #concatenate
        x2 = self.max_pool_2by2(x1)
        x3 = self.down_conv_2(x2) # concatenate
        x4 = self.max_pool_2by2(x3)
        x5 = self.down_conv_3(x4) # concatenate
        x6 = self.max_pool_2by2(x5)
        x7 = self.down_conv_4(x6) # concatenate
        x8 = self.max_pool_2by2(x7)
        x9 = self.down_conv_5(x8) 
        

        #decoding process
        x10 = self.up_process_1(x9)
        y = crop(x7,x10)
        x10 = self.up_conv_1(torch.cat([x10,y],1))
        
        x10 = self.up_process_2(x10)
        y = crop(x5,x10)
        x10 = self.up_conv_2(torch.cat([x10,y],1))

        x10 = self.up_process_3(x10)
        y = crop(x3,x10)
        x10 = self.up_conv_3(torch.cat([x10,y],1))

        x10 = self.up_process_4(x10)
        y = crop(x1,x10)
        x10 = self.up_conv_4(torch.cat([x10,y],1))
        print(x10.size())

        #output channel
        x10 = self.output_layer(x10)
        print(x10.size())
        # print(x10)
        return x10



        
      
if __name__ == "__main__":
    image = torch.rand((1,1,572,572))
    model = Unet()
    print(model.forward(image))


