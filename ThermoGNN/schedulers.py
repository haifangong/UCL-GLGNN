from math import cos, pi, sin


def linear(epoch, nepoch):
    return 1 - epoch / nepoch
# def linear(epoch, nepoch):
#     return epoch / nepoch
        # mask = max_probs.ge(p_cutoff * (class_acc[max_idx] / (2. - class_acc[max_idx]))).float()  # convex
        # mask = max_probs.ge(p_cutoff * (torch.log(class_acc[max_idx] + 1.) + 0.5)/(math.log(2) + 0.5)).float()  # concave
def convex(epoch, nepoch):
    x = epoch / nepoch
    # return cos((epoch / nepoch) * pi / 2)
    return epoch / (2-epoch)

def concave(epoch, nepoch):
    return 1 - sin((epoch / nepoch) * (pi / 2))


def composite(epoch, nepoch):
    return 0.5 * cos((epoch / nepoch) * pi) + 0.5
