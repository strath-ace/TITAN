import gpytorch as gpt

class SphericalMatern(gpt.kernels.Kernel):
    has_lengthscale = True

    def forward(self, x1, x2, **params):
        cos = x1 @ x2.T
        cos = cos.clamp(-1.0, 1.0)
        theta = torch.acos(cos)

        r = torch.sqrt(torch.tensor(3.0, device=x1.device))
        z = r * theta / self.lengthscale

        return (1.0 + z) * torch.exp(-z)

class SphericalGP(gpt.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpt.means.ConstantMean
        self.covar_module = gpt.kernels.ScaleKernel(SphericalMatern)

    def forward(self, x):
        mean_x = self.mean_module(x)
        cov = self.covar_module(x)
        return gpt.distributions.MultivariateNormal(mean_x, cov)
