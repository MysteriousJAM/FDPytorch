import json
import torch
import mxnet as mx

from collections import OrderedDict
from gluoncv.model_zoo.mobilenet import get_mobilenet

from utils.mobilenet import DetMobileNet


def load_mxnet_prams(arg_params, aux_params, net, allow_extra=False):
    names = [name for name, _ in net.named_parameters()]
    named_params = [(name, param) for name, param in net.named_parameters()]
    state_dict = OrderedDict()

    for index, mxnet_name in enumerate(arg_params):
        if allow_extra and index >= len(named_params):
            break
        mxnet_param = arg_params[mxnet_name].asnumpy()
        pytorch_name, pytorch_param = named_params[index]
        pytorch_param = pytorch_param.detach().numpy()
        if mxnet_param.shape != pytorch_param.shape:
            print(index)
            print(mxnet_param.shape, pytorch_param.shape)
            print(mxnet_name, pytorch_name)
            raise AssertionError
        state_dict[pytorch_name] = torch.from_numpy(mxnet_param)

    missing_keys = [p for p in net.state_dict()
                    if p not in names and 'num_batches_tracked' not in p]

    if not allow_extra:
        assert len(missing_keys) == len(aux_params)
    else:
        assert len(missing_keys) <= len(aux_params)
    for index, mxnet_name in enumerate(aux_params):
        if allow_extra and index >= len(named_params):
            break
        mxnet_param = aux_params[mxnet_name].asnumpy()
        state_dict[missing_keys[index]] = torch.from_numpy(mxnet_param)

    net.load_state_dict(state_dict, strict=False)
    net.eval()
    return net


if __name__ == '__main__':
    mb_net = get_mobilenet(multiplier=0.25)
    loaded = mx.nd.load('models/mobilenet0.25.params')
    params = mb_net._collect_params_with_prefix()
    for name in loaded:
        if name in params:
            ld = mx.nd.where(loaded[name].abs() >= 1, 1 * mx.nd.ones_like(loaded[name]) - 1e-6, loaded[name])
            params[name]._load_init(ld, ctx=mx.cpu(), cast_dtype=True, dtype_source='current')
    mb_net.hybridize()
    mb_net(mx.nd.zeros((1, 3, 512, 152)))
    mb_net.export('models/mobilenet0.25', 0)
    sym, arg_params, aux_params = mx.model.load_checkpoint('models/mobilenet0.25', 0)
    net = DetMobileNet(multiplier=0.25)
    net.eval()
    new_net = load_mxnet_prams(arg_params, aux_params, net, allow_extra=True)
    torch.save(new_net, 'models/mobilenet0.25.pth')
