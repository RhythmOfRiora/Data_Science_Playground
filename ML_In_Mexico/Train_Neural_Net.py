import torch
import time
import ML_In_Mexico.config.tools as tools
import ML_In_Mexico.Neural_Net as network
from torch import nn
import torch.nn.functional as F
import ML_In_Mexico.Covid_Dataset as cvd
import torch.optim as optim


def train_net():
    # Download and load the training data
    training_dataset = tools.load_pickle_file('../train.pckl')
    training_covid_dataset = cvd.CovidDataset(training_dataset, get_data=True, transform=None)
    training_dataloader = torch.utils.data.DataLoader(training_covid_dataset, batch_size=1, shuffle=True)

    validation_dataset = tools.load_pickle_file('../validation.pckl')
    validation_covid_dataset = cvd.CovidDataset(validation_dataset, get_data=True, transform=None)
    validation_dataloader = torch.utils.data.DataLoader(validation_covid_dataset, batch_size=1, shuffle=True)

    print(training_dataloader)
    print(training_dataloader)
    print(len(training_dataloader))

    net = network.Net()

    # Define the loss
    criterion = nn.CrossEntropyLoss()

    total_start_time = time.time()
    device = torch.device('cpu')

    # Optimizers require the parameters to optimize and a learning rate
    optimizer = optim.SGD(net.parameters(), lr=0.001)
    epochs = 5
    best_f1 = 0

    for e in range(epochs):
        running_loss = 0.0

        count = 0
        epoch_start_time = time.time()

        for i, data in enumerate(training_dataloader):

            net.train()

            input = data[0]
            label = data[1]

            input = input.type(torch.FloatTensor)
            input.requires_grad_(True)
            print("\n\nInput: ", input)

            optimizer.zero_grad()
            output = net(input)

            print("Output: ", output)
            print("Label: ", label)

            loss = criterion(input, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_end_time = time.time()
        print(epoch_end_time - epoch_start_time)

        print(f"Epoch done ... training loss: {running_loss / len(training_dataloader)}")


        # then we test it on the validation set

        precision, recall, f1, accuracy, fp_rate, miss_rate = test_network(net, validation_dataloader)

        # print the above out

        print("Precision: ", precision, "Recall: ", recall, "F1: ", f1, "Accuracy: ", accuracy, "False Positive Rate: ", fp_rate, "Miss Rate: ", miss_rate)
        if f1 > best_f1:
            best_metrics = [precision, recall, f1, accuracy, fp_rate, miss_rate]
            best_val_prec = precision
            best_val_recall = recall
            best_val_f1 = f1
            best_val_acc = accuracy
            best_val_fp_rate = fp_rate
            best_val_miss_rate = miss_rate

            # if the f1 score is a new best, set the best metrics, and save the model

        # then we test it on the train set

        precision, recall, f1, accuracy, fp_rate, miss_rate = test_network(net, training_dataloader)
        print("Testing on the training set: ")
        print("Precision: ", precision, "Recall: ", recall, "F1: ", f1, "Accuracy: ", accuracy, "False Positive Rate: ",
              fp_rate, "Miss Rate: ", miss_rate)

        # print the above out

    # once we get here, all the epochs are done. print out the best validation results.
    total_end_time = time.time()
    print("Total Time Taken To Train... ", tools.convert_time(total_end_time - total_start_time))

    return net, best_val_prec, best_val_recall, best_val_f1, best_val_acc, best_val_fp_rate, best_val_miss_rate, total_end_time

# save those returned results in a spreadsheet
# e.g.training
# DS
# path | validation
# DS
# path | model
# filename | input
# layer
# size | hidden1
# size | hidden2
# size | … |  # params | batch size | num epochs | training time | acc | precision | recall | f1 | fp% rate | miss% rate


def test_network(net=None, dataloader=None):
    """ Function to test a network. """
    # net = network.Net()

    # testing_dataset = tools.load_pickle_file('../testing.pckl')
    # testing_covid_dataset = cvd.CovidDataset(testing_dataset, get_data=True, transform=None)
    # testing_dataloader = torch.utils.data.DataLoader(testing_covid_dataset, batch_size=1, shuffle=True)

    # set the criterion
    count = 0
    test_loss = 0.0  # not the actual test set loss necessarily, just the loss related to whatever split is being handled in this function
    predictions = []
    target_labels = []

    with torch.no_grad():  # testing so no need to calculate gradients for backprop:
        for i, data in enumerate(dataloader, 0):

            count += 1
            net.eval()  # evaluation mode, not training anymore
            test_input = data[0]
            test_label = data[1]

            print(type(test_input))

            # test_input = [t.float() for t in test_input]
            test_input = test_input.type(torch.FloatTensor)

            print(test_input)
            # print(len(test_input))

            output = net(test_input)

            print(output)

            value = torch.round(output)
            predictions.append(value.type(torch.LongTensor))
            target_labels.append(test_label.type(torch.LongTensor))
            # # normalised_output = (torch.softmax(output, 1))
            # value = torch.max(output, 1)
            # predictions.append(index.item())
            # target_labels.append(test_label.item())


    print(f"Epoch done ... testing loss: {test_loss / len(dataloader)}")
    precision, recall, f1, accuracy, fp_rate, miss_rate = evaluate_network_performance(predictions, target_labels)

    # test
    # loss = test_loss / count
    # precision, recall, f1, accuracy, fp_rate, miss_rate = evaluate_network_performance(predictions, target_labels)
    #
    return precision, recall, f1, accuracy, fp_rate, miss_rate


def evaluate_network_performance(predictions, target_labels):
    # precision, recall, f1, accuracy, fp_rate, miss_rate
    print("Predictions: ", predictions)

    confusion_matrix = torch.zeros(2, 2)
    for i in range(len(predictions)):
        confusion_matrix[predictions[i], target_labels[i]] += 1

    print(confusion_matrix)
     # label 0 is non-intubed person (a negative) and label 1 is an intubed person (a positive i.e. what we are trying to detect)
     # [tn][fn]
     # [fp][tp]

    tn = confusion_matrix[0][0].item()
    fn = confusion_matrix[0][1].item()
    fp = confusion_matrix[1][0].item()
    tp = confusion_matrix[1][1].item()

    print(f'[tn] [fn]')
    print(f'[fp] [tp]')

    print(f'{int(tn)}  {int(fn)}')
    print(f'{int(fp)}  {int(tp)}')

    # print(f'tp = {tp}, fp = {fp}, tn = {tn}, fn = {fn}')

    # using -1 here handles ZeroDivisionError as it won’t be updated
    precision = -1
    recall = -1
    f1 = -1
    accuracy = -1
    fp_rate = -1
    miss_rate = -1

    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        print(f'precision: caught ZeroDivisionError')

    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        print(f'recall: caught ZeroDivisionError')
    if not precision == -1 and not recall == -1:
        try:
            f1 = 2 * ((precision * recall) / (precision + recall))
        except ZeroDivisionError:
            print(f'f1: caught ZeroDivisionError')  # shouldn't happen?
    try:
        accuracy = (tp + tn) / (tp + tn + fp + fn)
    except ZeroDivisionError:
        print(f'acc: caught ZeroDivisionError')  # shouldn't happen?
    try:
        fp_rate = 100 * fp / (fp + tn)
    except ZeroDivisionError:
        print(f'fp_rate: caught ZeroDivisionError')
    try:
        miss_rate = fn / (fp + tp)
    except ZeroDivisionError:
        print(f'miss_rate: caught ZeroDivisionError')

    return precision, recall, f1, accuracy, fp_rate, miss_rate

if __name__ == "__main__":
    # test_network()
    train_net()






