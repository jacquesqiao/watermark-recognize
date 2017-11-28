import paddle.v2 as paddle

import create_dataset
from resnet import resnet_cifar10

datadim = 200 * 200
classdim = 2


def main():

    # PaddlePaddle init
    paddle.init(use_gpu=True, trainer_count=1)

    image = paddle.layer.data(
        name="image", type=paddle.data_type.dense_vector(datadim))

    # Add neural network config
    # option 1. resnet
    net = resnet_cifar10(image, depth=32)
    # option 2. vgg
    # net = vgg_bn_drop(image)

    out = paddle.layer.fc(
        input=net, size=classdim, act=paddle.activation.Softmax())

    lbl = paddle.layer.data(
        name="label", type=paddle.data_type.integer_value(classdim))
    cost = paddle.layer.classification_cost(input=out, label=lbl)

    # load parameters
    with open("params_pass_167.tar", 'r') as f:
        parameters = paddle.parameters.Parameters.from_tar(f)

    for sample in create_dataset.test_reader()():
        data = sample[0]
        label = sample[1]
        probs = paddle.infer(output_layer=out,
                             parameters=parameters, input=[(data,)])
        prob = probs[0].tolist()
        infer_label = prob.index(max(prob))
        print(str(label) + " " + str(infer_label))


if __name__ == '__main__':
    main()
