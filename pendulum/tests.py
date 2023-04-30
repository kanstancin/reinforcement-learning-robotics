import math
import torch

from torch.distributions.normal import Normal
from torch.distributions import Categorical


def calc_pdf(action, mean, std):
    pdf = ((math.e ** (- ((action - mean) ** 2) / (2 * std ** 2))) / (std * math.sqrt(2 * math.pi)))
    return pdf


mean, std = 0, 0.2
distrib = Normal(mean, std)
action = distrib.sample()
prob = distrib.log_prob(action)

print(f'Normal: ')
print(f'action: {action}')
print(f'prob: {prob}')
print(math.e ** prob)
pdf = calc_pdf(action, mean, std)
print(f'Normal test:')
print(f'pdf: {pdf}')
print(f'pdf log: {math.log(pdf)}')
print(f'Mean of log_prob: {prob.mean()}')


print(f'\nCategorical:')
distrib = Categorical(torch.tensor([0.1, 0.5, 0.4]))
action = distrib.sample()
prob = distrib.log_prob(action)
print(f'action: {action}')
print(f'prob: {prob}')
print(math.e ** prob)



pdf = calc_pdf(action=0, mean=0, std=0.4)
print(f'\nPDF test:')
print(f'pdf: {pdf}')
print(f'pdf log: {-math.log(pdf)}')