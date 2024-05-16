'''
Reference: https://www.geeksforgeeks.org/generative-adversarial-network-gan/
'''
import tensorflow as tf

class DCGAN_generator(tf.Module):
  """
  Attributes
  ----------
    ngpu : int
      The number of available GPU devices
  """
  def __init__(self, ngpu):
    """Init function
    Parameters
    ----------
      ngpu : int
        The number of available GPU devices
    """
    super(DCGAN_generator, self).__init__()
    self.ngpu = ngpu
    
    nz = 100 # noise dimension
    ngf = 64 # number of features map on the first layer
    nc = 3 # number of channels

    self.main = nn.Sequential(
      # input is Z, going into a convolution
      nn.ConvTranspose2d(     nz, ngf * 4, 4, 1, 0, bias=False),
      nn.BatchNorm2d(ngf * 4),
      nn.ReLU(True),
      # state size. (ngf*8) x 4 x 4
      nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ngf * 2),
      nn.ReLU(True),
      # state size. (ngf*4) x 8 x 8
      nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ngf),
      nn.ReLU(True),
      # state size. (ngf*2) x 16 x 16
      nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
      nn.Tanh()
      # state size. (nc) x 64 x 64
    )

  def forward(self, input):
    """Forward function
    Parameters
    ----------
    input : :py:class:`torch.Tensor`
    
    Returns
    -------
    :py:class:`torch.Tensor`
      the output of the generator (i.e. an image)
    """
    output = self.main(input)
    return output