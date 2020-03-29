import numpy as np
import Model as mdl


class Pa(mdl.Model):
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

            # Checks if the prediction is right
            if label != predicted_label:

                # tao = max(0, ((1 - np.dot(data[index],
                # self.weights[lable_index, :]) + np.dot(data[index], self.weights[
                # predicted_label, :])))) / (2 * (np.linalg.norm(data[index]) ** 2))
                wyx = np.dot(self.weights[label], data[index])
                wy_hat_x = np.dot(self.weights[predicted_label], data[index])
                loss = max(0, 1 - wyx + wy_hat_x)
                norm = (2 * (np.linalg.norm(data[index]) ** 2))
                tao = loss / norm
                self.weights[label] = self.weights[label] + tao * data[index]
                self.weights[predicted_label] = self.weights[predicted_label] - tao * data[index]

