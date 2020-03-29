import numpy as np
import Model as mdl


class Svm(mdl.Model):
    def __init__(self, train_x, eta, epochs, ingore_vacc, lamda, num_of_classes=3):
        mdl.Model.__init__(self, train_x, eta, epochs, ingore_vacc, num_of_classes)
        self.lamda = lamda
        self.details_to_print.append("lambda = " + str(self.lamda))

    def iteration(self, data, labels):

        data = data.astype(float)
        labels = labels.astype(float)

        # ignore the features we want to ignore
        for i in range(data.shape[0]):
            data[i] = np.multiply(data[i], self.ingore_vacc)

        for index in range(len(data)):
            predicted_label = self.predict(data[index])
            label = labels[index, 0].astype(int)

            if label != predicted_label:
                for i in range(0, self.weights.shape[0]):
                    # If the current class is the correct class
                    if i == label:
                        # w(t+1) = w(t)*(1 - eta * lamda)
                        self.weights[i] = np.add(np.multiply(self.weights[i], 1 - self.eta * self.lamda),
                                                 np.multiply(data[index], self.eta))
                    elif i == predicted_label:
                        # w(t + 1) = w(t)*(1 - eta * lamda) - x * eta
                        self.weights[i] = np.subtract(np.multiply(self.weights[i], 1 - self.eta * self.lamda),
                                                      np.multiply(data[index], self.eta))
                    else:
                        # w(t+1) = w(t)*(1 - eta * lamda)
                        self.weights[i] = np.multiply(self.weights[i], 1 - self.eta * self.lamda)
                # TODO to check if this does the same as the for loop above
                # TODO w(t+1) = w(t)*(1 - eta * lamda)
                # TODO self.weights = np.multiply(self.weights, 1 - self.eta * self.lamda)
                # TODO eta_x_train = np.multiply(data_instance, self.eta)
                # TODO self.weights[lable_index] = np.add(self.weights[label_instance], eta_x_train)
                # TODO self.weights[predicted_label] = np.subtract(self.weights[label_instance], eta_x_train)
            else:
                for i in range(0, self.weights.shape[0]):
                    # w(t+1) = w(t)*(1 - eta * lamda)
                    self.weights[i] = np.multiply(self.weights[i], 1 - self.eta * self.lamda)

                # TODO to check if this does the same as the for loop above
                # w(t+1) = w(t)*(1 - eta * lamda)
                # TODO self.weights = np.multiply(self.weights, 1 - self.eta * self.lamda)

    def add_hyper_parameters_to_log(self):

        self.logging_information.append("\n#############################")
        self.logging_information.append("Number of epochs = " + str(self.epochs))
        self.logging_information.append("Start eta = " + str(self.eta))
        self.logging_information.append("lambda = " + str(self.lamda))
