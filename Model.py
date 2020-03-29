import numpy as np


class Model(object):
    def __init__(self, train_x, eta, epochs, ingore_vacc, num_of_classes=3):
        # We added the +1 for bias, we treat it as one of our weights.
        self.weights = np.zeros((num_of_classes, train_x.shape[1])) # 3 rows 8 columns
        self.weights[:, -1] = 1
        self.eta = eta
        self.epochs = epochs
        self.ingore_vacc = ingore_vacc
        self.details_to_print = []
        self.logging_information = []

    def train(self, data, labels, eta, epoch_num, k_fold_parameter=1):
        self.eta = eta
        self.epochs = epoch_num
        self.add_hyper_parameters_to_log()

        # Shuffle the data
        data, labels = self.shuffle_data(data, labels)

        # Accuracy variable
        avg_accuracy = 0

        save_eta = self.eta

        for i in range(k_fold_parameter):
            # Initiate the weights
            self.weights = np.zeros(self.weights.shape)  # 3 rows 8 columns
            self.weights[:, -1] = 1

            self.eta = save_eta

            data_x, data_y, test_x, test_y = self.k_fold_data_arrnge(data, labels, k_fold_parameter, i)

            for epoch in range(self.epochs):
                self.iteration(data_x, data_y)

                if k_fold_parameter != 1:
                    accuracy_total, accuracy_test, accuracy_train = self.calc_accuracy(data_x, data_y, test_x, test_y)

                    # Add details to the report
                    self.details_to_print.append("epoch number = " + str(epoch))
                    self.details_to_print.append("eta = " + str(self.eta))

                    #self.details_to_print.append("wights = " + str(self.weights))
                    self.details_to_print.append("accuracy_total = " + str(accuracy_total))
                    self.details_to_print.append("accuracy_train  = " + str(accuracy_train))
                    self.details_to_print.append("accuracy_test = " + str(accuracy_test))

                # Update the eta
                self.eta = self.eta / (epoch + 1)

            self.eta = save_eta

            # If this is not testing mode
            if k_fold_parameter == 1:
                return 0


            # Check the accuracy of the current k fold test part
            accuracy_total, accuracy_test, accuracy_train = self.calc_accuracy(data_x, data_y, test_x, test_y)
            avg_accuracy = avg_accuracy + accuracy_test

            # Write information about the current k fold test data
            self.logging_information.append("For test part: " + str(i))
            self.logging_information.append("accuracy_total = " + str(accuracy_total))
            self.logging_information.append("accuracy_train  = " + str(accuracy_train))
            self.logging_information.append("accuracy_test = " + str(accuracy_test))
            self.logging_information.append("\n")
            self.logging_information.append("\n")

        avg_accuracy = avg_accuracy / k_fold_parameter
        self.logging_information.append("Average accuracy in k fold= " + str(avg_accuracy))
        self.logging_information.append("##########################\n" + str(avg_accuracy))

        return avg_accuracy

    def predict(self, data):
        return np.argmax(np.dot(self.weights, data))

    def iteration(self, data, labels):
        i = 1

    def calc_accuracy(self, train_x, train_y, test_x, test_y):
        accuracy_train = 0
        accuracy_test = 0
        counter_train = 0
        counter_test = 0

        train_x = train_x.astype(float)
        train_y = train_y.astype(float)
        test_x = test_x.astype(float)
        test_y = test_y.astype(float)

        # ignore the features we want to ignore
        for i in range(train_x.shape[0]):
            train_x[i] = np.multiply(train_x[i], self.ingore_vacc)

        for i in range(test_x.shape[0]):
            test_x[i] = np.multiply(test_x[i], self.ingore_vacc)

        # Calc the accuracy on the data set
        for data_instance, label_instance in zip(train_x, train_y):
            predicted_label = self.predict(data_instance)

            if label_instance == predicted_label:
                counter_train += 1

        if len(train_x) != 0:
            accuracy_train = counter_train / len(train_x)

        for data_instance, label_instance in zip(test_x, test_y):
            predicted_label = self.predict(data_instance)

            if label_instance == predicted_label:
                counter_test += 1

        if len(test_x) != 0:
            accuracy_test = counter_test / len(test_x)

        counter_total = counter_train + counter_test
        if len(test_x) + len(train_x) != 0:
            accuracy_total = counter_total / (len(test_x) + len(train_x))

        return accuracy_total, accuracy_test, accuracy_train

    def print_details(self):
        #for line in self.details_to_print:
        #    print(line)

        #print("\n\n")

        for line in self.logging_information:
            print(line)

    def shuffle_data(self, data, labels):

        # Shuffle the data
        zipdata = list(zip(data, labels))
        np.random.shuffle(zipdata)
        data, labels = zip(*zipdata)

        data = np.asarray(data)
        data = data.astype(float)
        labels = np.asarray(labels)
        labels = labels.astype(float)

        return data, labels

    def k_fold_data_arrnge(self, data, labels, number_of_parts, test_part_number):
        size = len(data)
        if number_of_parts == 1:
            data_x = data
            data_y = labels
            test_x = np.zeros(data.shape[1])
            test_y = np.zeros(labels.shape[1])
        elif test_part_number == 0:
            test_x = data[0: int(size / number_of_parts)]
            test_y = labels[0: int(size / number_of_parts)]
            data_x = data[int(size / number_of_parts): size]
            data_y = labels[int(size / number_of_parts): size]
        elif test_part_number == number_of_parts - 1:
            test_x = data[int(size * (number_of_parts - 1) / number_of_parts): size]
            test_y = labels[int(size * (number_of_parts - 1) / number_of_parts): size]
            data_x = data[0: int(size * (number_of_parts - 1) / number_of_parts)]
            data_y = labels[0:int(size * (number_of_parts - 1) / number_of_parts)]
        else:
            test_x = data[int(size * test_part_number / number_of_parts):
                          int(size * (test_part_number + 1) / number_of_parts)]
            test_y = labels[int(size * test_part_number / number_of_parts):
                            int(size * (test_part_number + 1) / number_of_parts)]

            pretest_x = data[0: int(size * test_part_number / number_of_parts)]
            posttest_x = data[int(size * (test_part_number + 1) / number_of_parts): size]
            pretest_y = labels[0: int(size * test_part_number / number_of_parts)]
            posttest_y = labels[int(size * (test_part_number + 1) / number_of_parts): size]

            data_x = np.concatenate((pretest_x, posttest_x), 0)
            data_y = np.concatenate((pretest_y, posttest_y), 0)

        return data_x, data_y, test_x, test_y

    def add_hyper_parameters_to_log(self):
        self.logging_information.append("--------------------------------")
        self.logging_information.append("Number of epochs = " + str(self.epochs))
        self.logging_information.append("Start eta = " + str(self.eta))
