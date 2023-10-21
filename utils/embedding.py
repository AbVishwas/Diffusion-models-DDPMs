import torch


def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    #n = number of timesteps
    #d = dimension of embedding
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])   #calculates a set of frequencies for the sinusoidal functions, values that decay exponentially as j increases.
                                                                       #These frequencies are used to generate the sine and cosine components of the embeddings

    wk = wk.reshape((1, d))               #This is done to ensure that wk has the correct shape for later element-wise multiplication with the positional indices.

    t = torch.arange(n).reshape((n, 1))

    embedding[:,::2] = torch.sin(t * wk[:,::2])             # sine components of the embeddings for even-indexed positions
    embedding[:,1::2] = torch.cos(t * wk[:,::2])            #  cosine components of the embeddings for odd-indexed positions

    return embedding