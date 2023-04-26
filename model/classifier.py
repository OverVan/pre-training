from torch.nn import Module, Linear

from .model import make, register


@register("classifier")
class Classifier(Module):
    def __init__(self, encoder, cls_num):
        super(Classifier, self).__init__()
        self.encoder = make(encoder["name"], **encoder["args"])
        self.linear_classifier = LinearClassifier(self.encoder.out_dim, cls_num)

    def forward(self, x):
        x = self.encoder(x)
        x = self.linear_classifier(x)
        return x


class LinearClassifier(Module):
    def __init__(self, in_dim, cls_num):
        super(LinearClassifier, self).__init__()
        self.linear = Linear(in_dim, cls_num)

    def forward(self, x):
        return self.linear(x)
