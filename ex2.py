import SVM as svm
import Perceptron as prtn
import Pa as pa
import FileReader as flr
import DataManipulation as dm
import numpy as np
import sys


def hyper_parameters_testing():
    data = flr.read_from_file("train_x.txt")
    data = dm.add_bias(data)
    data = dm.convert_sex_to_number(data, 1, 1, 0)
    data = np.array(data)

    lables = flr.read_from_file("train_y.txt")
    lables = np.array(lables)
    lables = dm.convert_to_float(lables)

    # Normalaise the data
    min_range = np.ones(data.shape[1])
    min_range = np.multiply(-1, min_range)
    max_range = np.ones(data.shape[1])
    data = dm.min_max_normalization(data, min_range, max_range)

    # Set the properties we want to use
    ignore = np.ones(len(data[0]))

    # Pa algorithem
    alg_pa = pa.Pa(data, 0.1, 100, ignore, 3)
    alg_pa.train(data, lables, 0.01, 26, 5)
    alg_pa.print_details()

    eta_list = []
    epocnum_list = []
    accuracy_list = []

    # Perceptron algorithem
    #alg = prtn.Perceptron(data, 0.1, 100, ignore, 3)
    #accuracy = alg.train(data, lables, 0.01, 22, 5)

    #eta_list.append(0.01)
    #epocnum_list.append(20)
    #accuracy_list.append(accuracy)

    #eta = 1
    #for index_eta in range(5):
    #    epoch = 5
    #    for index_epoch in range(5):
    #        accuracy = alg.train(data, lables, eta, epoch, 4)

    #        eta_list.append(eta)
    #        epocnum_list.append(epoch)
    #        accuracy_list.append(accuracy)

            # Update the hyper parameters
     #       epoch = 2 * epoch
     #   eta = 0.1 * eta

    #print("Perceptron")
    #alg.print_details()

    #print(eta_list)
    #print(epocnum_list)
    #print(accuracy_list)
    #print("****************************")

    # Perceptron algorithem
    #print("perseptrpn")
    #alg = prtn.Perceptron(data, 0.1, 100, ignore, 3)
    #accuracy = alg.train(data, lables, 0.01, 20, 5)
    #alg.print_details()
    #print(accuracy)


    # SVM algorithem
    #print("svm")
    #alg = svm.Svm(data, 0.01, 100, ignore, 0.001, 3)
    #alg.lamda = 0.1
    #accuracy = alg.train(data, lables, 0.1, 20, 4)
    #alg.print_details()
    #print(accuracy)

 #   eta = 1
 #   for index_eta in range(5):
 #       epoch = 1
 #       for index_epoch in range(5):
 #           lamda = 0.2
 #           for index_lamda in range(1):
 #               alg.lamda = lamda
 #               accuracy = alg.train(data, lables, eta, epoch, 4)
"""
                eta_list.append(eta)
                epocnum_list.append(epoch)
                accuracy_list.append(accuracy)
                print("bla")
                # Update the hyper parameters
                #amda = lamda * 2

            epoch = 1 + epoch
        eta = 0.1 * eta
"""
    #print("SVM")
    #alg.print_details()

    #print(eta_list)
    #print(epocnum_list)
    #print(accuracy_list)

  #  # Pa algorithem
  #  alg = pa.Pa(data, 0.1, 100, ignore, 3)
  #  epoch = 5

  #  for index_eta in range(2):
  #      eta = 1
  #      for index_epoch in range(5):
  #          accuracy = alg.train(data, lables, eta, epoch)

   #         eta_list.append(eta)
   #         epocnum_list.append(epoch)
   #         accuracy_list.append(accuracy)

            # Update the hyper parameters
    #        eta = 0.1 * eta
     #   epoch = 2 * epoch

    #print("PA")
    #alg.print_details()

    #print(eta_list)
    #print(epocnum_list)
    #print(accuracy_list)


def final_format(data_file, lables_file, test_file):
    # Getting the data and convert it to be useable by the algorthems
    data = flr.read_from_file(data_file)
    data = dm.add_bias(data)
    data = dm.convert_sex_to_number(data, 1, 1, 0)
    data = np.array(data)

    # Normalaize the data
    min_range = np.ones(data.shape[1])
    min_range = np.multiply(-1, min_range)
    max_range = np.ones(data.shape[1])
    data = dm.min_max_normalization(data, min_range, max_range)

    # Getting the lables and convert it to be useable by the algorthems
    lables = flr.read_from_file(lables_file)
    lables = np.array(lables)
    lables = dm.convert_to_float(lables)

    # Get the test data and convert it to be useable by the algorthems
    test = flr.read_from_file(test_file)
    test = dm.add_bias(test)
    test = dm.convert_sex_to_number(test, 1, 1, 0)
    test = np.array(test)

    # Normalaize the test data
    min_range = np.ones(data.shape[1])
    min_range = np.multiply(-1, min_range)
    max_range = np.ones(data.shape[1])
    test = dm.min_max_normalization(test, min_range, max_range)

    # Set the properties we want the algorithems to use to use
    ignore = np.ones(len(data[0]))

    # Perceptron algorithem
    alg_peceptron = prtn.Perceptron(data, 0.1, 100, ignore, 3)
    alg_peceptron.train(data, lables, 0.01, 20)

    # Svm algorithem
    alg_svm = svm.Svm(data, 0.01, 100, ignore, 0.001, 3)
    alg_svm.lamda = 0.1
    alg_svm.train(data, lables, 0.1, 20)

    # Pa algorithem
    alg_pa = pa.Pa(data, 0.1, 100, ignore, 3)
    alg_pa.train(data, lables, 0.01, 25)

    # Compare the algoritems on the test
    for test_data in test:
        line_to_print = "perceptron: " + str(alg_peceptron.predict(test_data)) + ", "
        line_to_print += "svm: " + str(alg_svm.predict(test_data)) + ", "
        line_to_print += "pa: " + str(alg_pa.predict(test_data))
        print(line_to_print)


np.random.seed(10)
final_format(sys.argv[1], sys.argv[2], sys.argv[3])
#hyper_parameters_testing()
