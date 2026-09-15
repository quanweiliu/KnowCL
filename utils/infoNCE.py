import torch
import torch.nn.functional as F


def sim(z1: torch.Tensor, z2: torch.Tensor):
    return torch.mm(z1, z2.t())

def semi_loss(z1: torch.Tensor, z2: torch.Tensor, tau=0.2):

    z1, z2 = F.normalize(z1), F.normalize(z2)

    f = lambda x: torch.exp(x / tau)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))

    return (-torch.log(between_sim.diag()
        / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))).mean()

if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    batchSize = 8
    feature_dim = 128
    feature_1, out_1 = torch.rand(batchSize,feature_dim), torch.rand(batchSize,feature_dim)
    feature_2, out_2 = torch.rand(batchSize,feature_dim), torch.rand(batchSize,feature_dim)
    loss1 = semi_loss(out_1, out_2)
    loss2 = semi_loss(out_2, out_1)
    loss = (loss1 + loss2) * 0.5
    print(loss)