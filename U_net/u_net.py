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

    def forward(self, image):
        """
        take an image and run the forward pass
        """
        x1 = self.down_conv_1(image) 
        print(x1.size())
        x2 = self.max_pool_2by2(x1)
        x3 = self.down_conv_2(x2) 
        x4 = self.max_pool_2by2(x3)
        x5 = self.down_conv_3(x4) 
        x6 = self.max_pool_2by2(x5)
        x7 = self.down_conv_4(x6) 
        x8 = self.max_pool_2by2(x7)
        x9 = self.down_conv_5(x8) 
        print(x9.size())
   
      
if __name__ == "__main__":
    image = torch.rand((1,1,572,572))
    model = Unet()
    print(model.forward(image))

