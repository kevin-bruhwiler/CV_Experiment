from keras.applications import Xception
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed


def make_models():
    validator = Xception(weights="imagenet", include_top=True)
    validator.trainable = False
    for layer in validator.layers:
        layer.trainable = False

    predictor_input = Xception(weights="imagenet", include_top=True)
    predictor_input.layers.pop()
    predictor_input.trainable = False
    for layer in predictor_input.layers:
        layer.trainable = False

    predictor = Sequential()
    inp_shape = validator.layers[0].input_shape
    predictor.add(TimeDistributed(predictor_input, input_shape=(30,inp_shape[1],inp_shape[2],inp_shape[3])))
    predictor.add(LSTM(2048, return_sequences=True))
    predictor.add(LSTM(2048))
    predictor.add(validator.layers[-1])
    predictor.layers[0].trainable = False

    return predictor, validator


def imagenet_outputs():
    with open("imagenet_outputs", "rb") as file:
        dicts = eval(file.read())
    return dicts


def run():
    predictor, validator = make_models()
    output_translator = imagenet_outputs()


if __name__=="__main__":
    run()