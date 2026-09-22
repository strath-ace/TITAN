import gpytorch as gpt
import torch
import numpy as np

class ChordalKernel(gpt.kernels.Kernel):
    has_lengthscale = True

    def forward(self, x1, x2, **params):
        x1 = torch.nn.functional.normalize(x1, dim=-1)
        x2 = torch.nn.functional.normalize(x2, dim=-1)

        cosine = (x1 @ x2.transpose(-1, -2)).clamp(-1.0, 1.0)

        # Squared chordal distance is 2 * (1 - cosine).
        distance_squared = 2.0 * (1.0 - cosine)

        return torch.exp(
            -0.5 * distance_squared / self.lengthscale.square()
        )

class SphericalGP(gpt.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpt.means.ConstantMean()
        self.covar_module = gpt.kernels.ScaleKernel(ChordalKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        cov = self.covar_module(x)
        return gpt.distributions.MultivariateNormal(mean_x, cov)

class MultiOutputGP():
    '''Container class for training and holding multiple GPs for multivariate outputs'''
    def __init__(self, x : torch.Tensor, y : torch.Tensor, gpt_model : gpt.models.ExactGP, likelihood : gpt.likelihoods.Likelihood, train=True, training_iters=50, learning_rate = 0.1):
        self.n_train = training_iters
        self.n_inputs = x.size()[1]
        self.n_outputs = y.size()[1]

        self.likelihoods = []
        self.models = []
        self.learning_rate = learning_rate
        for i_output in range(self.n_outputs):
            self.likelihoods.append(likelihood())
            self.models.append(gpt_model(x, y[:,i_output], self.likelihoods[-1]))
        if train: self.train(x, y)

        

    def train(self, x : torch.Tensor, y : torch.Tensor):
        for i_output in range(self.n_outputs):
            likelihood = self.likelihoods[i_output]
            model = self.models[i_output]

            model.train()
            likelihood.train()

            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            mll = gpt.mlls.ExactMarginalLogLikelihood(likelihood, model)
            for i in range(self.n_train):
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = model(x)
                # Calc loss and backprop gradients
                loss = -mll(output, y[:,i_output])
                loss.backward()
                print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                    i + 1, self.n_train, loss.item(),
                    model.covar_module.base_kernel.lengthscale.item(),
                    model.likelihood.noise.item()
                ))
                optimizer.step()

    def evaluate(self, x):
        if not isinstance(x ,torch.Tensor): x = torch.tensor(x)
        x = torch.atleast_2d(x)
        output = np.zeros(self.n_outputs)
        for i_output in range(self.n_outputs):
            self.models[i_output].eval()
            self.likelihoods[i_output].eval()

            output[i_output] = float(self.models[i_output](x).mean.detach())
        return output

    __call__ = evaluate