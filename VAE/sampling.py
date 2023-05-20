from traineval import *

checkpoint = torch.load('base')
generated_samples = checkpoint['samples']
print(generated_samples.shape)

print(generated_samples[0])
