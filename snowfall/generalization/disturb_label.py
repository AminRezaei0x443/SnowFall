import torch
from torch import nn
from torch import distributions


class DisturbLabel(nn.Module):
    def __init__(self, alpha, class_count):
        """
        DistrubLabel Regularization
        :param alpha: probability of label disturbing , float [0,1]
        :param class_count: target space classes count
        """
        super(DisturbLabel, self).__init__()
        self.alpha = alpha
        self.class_count = class_count
        # Multinoulli distribution
        self.p_c = (1 - ((class_count - 1)/class_count) * alpha)
        self.p_i = (1 / class_count) * alpha

    def forward(self, y):
        # convert classes to index
        y_tensor = y
        y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)

        # create disturbed labels
        depth = self.class_count
        y_one_hot = torch.ones(y_tensor.size()[0], depth) * self.p_i
        y_one_hot.scatter_(1, y_tensor, self.p_c)
        y_one_hot = y_one_hot.view(*(tuple(y.shape) + (-1,)))

        # sample from Multinoulli distribution
        distribution = distributions.OneHotCategorical(y_one_hot)
        y_disturbed = distribution.sample()
        y_disturbed = y_disturbed.max(dim=1)[1]  # back to categorical

        return y_disturbed
