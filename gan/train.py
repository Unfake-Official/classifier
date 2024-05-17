from gan.discriminator_v1 import DCGAN_generator
from gan.generator_v1 import 

# data for plotting purposes
generatorLosses = []
discriminatorLosses = []
classifierLosses = []

#training starts

epochs = 100

# models
netG = DCGAN_generator(1)
netD = DCGAN_discriminator(1)
netC = ResNet18()

netG.to(device)
netD.to(device)
netC.to(device)

# optimizers 
optD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay = 1e-3)
optG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
optC = optim.Adam(netC.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay = 1e-3)

advWeight = 0.1 # adversarial weight

loss = nn.BCELoss()
criterion = nn.CrossEntropyLoss()