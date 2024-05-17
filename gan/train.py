from dcgan_generator import DCGAN_Generator
from dcgan_discriminator import DCGAN_Discriminator
from resnet import ResidualBlock, ResNet

# data for plotting purposes
generatorLosses = []
discriminatorLosses = []
classifierLosses = []

epochs = 100

netG = DCGAN_Generator(1)
netD = DCGAN_Discriminator(1)
netC = ResNet(ResidualBlock, [2, 2, 2, 2])

# optimizers
optD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay = 1e-3)
optG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
optC = optim.Adam(netC.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay = 1e-3)

advWeight = 0.1 # adversarial weight

loss = nn.BCELoss()
criterion = nn.CrossEntropyLoss()

def train(datasetLoader):
  file.write(text)
  for epoch in range(epochs):
  netC.train()

  running_loss = 0.0
  total_train = 0
  correct_train = 0
  for i, data in enumerate(subTrainLoader, 0):

    dataiter = iter(subTrainLoader)
    inputs, labels = dataiter.next()
    inputs, labels = inputs.to(device), labels.to(device)
    tmpBatchSize = len(labels)

    # create label arrays
    true_label = torch.ones(tmpBatchSize, 1, device=device)
    fake_label = torch.zeros(tmpBatchSize, 1, device=device)

    r = torch.randn(tmpBatchSize, 100, 1, 1, device=device)
    fakeImageBatch = netG(r)

    real_cpu = data[0].to(device)
    batch_size = real_cpu.size(0)

    # train discriminator on real images
    predictionsReal = netD(inputs)
    lossDiscriminator = loss(predictionsReal, true_label) #labels = 1
    lossDiscriminator.backward(retain_graph = True)

    # train discriminator on fake images
    predictionsFake = netD(fakeImageBatch)
    lossFake = loss(predictionsFake, fake_label) #labels = 0
    lossFake.backward(retain_graph= True)
    optD.step() # update discriminator parameters

    # train generator
    optG.zero_grad()
    predictionsFake = netD(fakeImageBatch)
    lossGenerator = loss(predictionsFake, true_label) #labels = 1
    lossGenerator.backward(retain_graph = True)
    optG.step()

    torch.autograd.set_detect_anomaly(True)
    fakeImageBatch = fakeImageBatch.detach().clone()

    # train classifier on real data
    predictions = netC(inputs)
    realClassifierLoss = criterion(predictions, labels)
    realClassifierLoss.backward(retain_graph=True)

    optC.step()
    optC.zero_grad()

    # update the classifer on fake data
    predictionsFake = netC(fakeImageBatch)
    # get a tensor of the labels that are most likely according to model
    predictedLabels = torch.argmax(predictionsFake, 1) # -> [0 , 5, 9, 3, ...]
    confidenceThresh = .2

    # psuedo labeling threshold
    probs = F.softmax(predictionsFake, dim=1)
    mostLikelyProbs = np.asarray([probs[i, predictedLabels[i]].item() for  i in range(len(probs))])
    toKeep = mostLikelyProbs > confidenceThresh
    if sum(toKeep) != 0:
        fakeClassifierLoss = criterion(predictionsFake[toKeep], predictedLabels[toKeep]) * advWeight
        fakeClassifierLoss.backward()

    optC.step()

    # reset the gradients
    optD.zero_grad()
    optG.zero_grad()
    optC.zero_grad()

    # save losses for graphing
    generatorLosses.append(lossGenerator.item())
    discriminatorLosses.append(lossDiscriminator.item())
    classifierLosses.append(realClassifierLoss.item())

    # get train accurcy
    if(i % 100 == 0):
      netC.eval()
      # accuracy
      _, predicted = torch.max(predictions, 1)
      total_train += labels.size(0)
      correct_train += predicted.eq(labels.data).sum().item()
      train_accuracy = 100 * correct_train / total_train
      text = ("Train Accuracy: " + str(train_accuracy))
      file.write(text + '\n')
      netC.train()

  print("Epoch " + str(epoch) + "Complete")

  # save gan image
  gridOfFakeImages = torchvision.utils.make_grid(fakeImageBatch.cpu())
  torchvision.utils.save_image(gridOfFakeImages, "/content/gridOfFakeImages/" + str(epoch) + '_' + str(i) + '.png')
  validate()
