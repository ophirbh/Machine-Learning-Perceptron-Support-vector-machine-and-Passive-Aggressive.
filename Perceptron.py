import numpy as np
import Model as mdl


class Perceptron(mdl.Model):
    def __init__(self, train_x, eta, epochs, ingore_vacc, num_of_classes):
        mdl.Model.__init__(self, train_x, eta, epochs, ingore_vacc, num_of_classes)

    def iteration(self, data, labels):
        # Rearrange the lable vecc for conteble reasons
        data = data.astype(float)
        labels = labels.astype(float)

        # ignore the features we want to ignore
        for i in range(data.shape[0]):
            data[i] = np.multiply(data[i], self.ingore_vacc)

        for index in range(len(data)):
            predicted_label = self.predict(data[index])
            label = labels[index, 0].astype(int)

            if label != predicted_label:
                self.weights[label, :] = self.weights[label, :] + self.eta * data[index]
                self.weights[predicted_label, :] = self.weights[predicted_label, :] - self.eta * data[index]
