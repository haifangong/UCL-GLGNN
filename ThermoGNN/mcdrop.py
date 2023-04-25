import argparse

import numpy as np
import torch


class MCDrop(object):
    def __init__(self, model, times=5):
        model.eval()
        self.times = times
        self.model = model
        # self.model.cuda()
        self.enable_dropout(self.model)

    def predict(self, _input):
        var_list = []
        for _ in range(self.times):
            pred = self.model(_input)
            # print('pred', pred)
            pred = pred.cpu().detach().numpy()
            # print('shape', pred.shape)
            var_list.append(pred)
        var = np.var(np.array(var_list), axis=1)
        # print(np.mean(var))
        return np.round(np.mean(var), 3)

    def enable_dropout(self, model):
        """ Function to enable the dropout layers during test-time """
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()


class Predictor(object):
    def __init__(self, model, weights):
        torch.cuda.is_available()
        # print("GPU {}".format(use_gpu))
        # print("GPU count {}".format(torch.cuda.device_count()))
        device_index = torch.cuda.current_device()
        print("Using {}".format(torch.cuda.get_device_name(device_index)))
        print("__init__Predictor")
        print("Input model is \n {}".format(model))
        print("Loading trained weights...")
        weights = './weights/' + weights
        model.load_state_dict(torch.load(weights))
        print("Done.")
        model.eval()

        self.model = model
        self.model.cuda()
        self.enable_dropout(self.model)

    def predict(self, _input, mode='with_uncertainty', n_samples=10):
        _input = np.reshape(np.array(_input), (-1, 1))
        _input = torch.Tensor(_input).cuda()

        if mode == 'without_uncertainty':
            pred = self.model(_input)
            pred = pred.cpu().detach().numpy()
            print(pred)

        elif mode == 'with_uncertainty':
            result = []

            for _ in range(n_samples):
                pred = self.model(_input)
                pred = pred.cpu().detach().numpy()
                result.append(pred[0][0])

            var = np.var(np.array(result))
            return var

    def enable_dropout(self, model):
        """ Function to enable the dropout layers during test-time """
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shape', default='curve', type=str)
    args = parser.parse_args()

    if args.shape == 'curve':
        model = 'xxx'
        trained_weights = str(args.shape) + '.pth'
        predictor = Predictor(model, trained_weights)
        print('Start predicting ...')
        predictor.predict(np.pi, mode='with_uncertainty')
        print('Done.')
