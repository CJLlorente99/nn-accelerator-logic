from torch.nn.utils import prune
import random
import math

def random_pruning_per_neuron(module, name, connectionsToPrune):

    class RandomPruningPerNeuron(prune.BasePruningMethod):
        PRUNING_TYPE = 'unstructured'

        def compute_mask(self, t, default_mask):
            mask = default_mask.clone()
            for i in range(mask.shape[0]):
                randomlist = random.sample(range(mask.shape[1]), connectionsToPrune)
                mask[i, randomlist] = 0
            return mask
    
    RandomPruningPerNeuron.apply(module, name)
    return module

def ordered_pruning_per_neuron(module, name, connectionsToPrune):

    class OrderedPruningPerNeuron(prune.BasePruningMethod):
        PRUNING_TYPE = 'unstructured'

        def compute_mask(self, t, default_mask):
            mask = default_mask.clone()
            for i in range(mask.shape[0]):
                randomlist = [j % mask.shape[1] for j in range(i * math.ceil(mask.shape[1] / 100) + (mask.shape[1] - connectionsToPrune), i * math.ceil(mask.shape[1] / 100) +  mask.shape[1])]
                mask[i, randomlist] = 0
            return mask
    
    OrderedPruningPerNeuron.apply(module, name)
    return module