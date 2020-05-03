import csv
import matplotlib.pyplot as plt
import os

"""
Get all filepaths in given folder
"""
def get_filepaths(dir):
    files = []
    for p in os.listdir(dir):
        path = os.path.join(dir, p)
        if os.path.isfile(path):
            files.append([path, p])
    return files


"""Generates the plot of one training history file"""
def generate_plots(file, name):
    with open(file, "r") as infile:
        reader = csv.reader(infile)
        real = []
        predicted = []
        for row in reader:
            if not row[0] == "Real":
                real.append(float(row[0]))
                predicted.append(float(row[1]))

    plt.plot(real[0:200])
    plt.plot(predicted[0:200])
    plt.ylabel("Signal value")
    plt.xlabel("Step")
    plt.xlim(0, 200)
    plt.ylim(-15, 15)
    plt.title('Real versus predicted signal')
    plt.legend(('Real', 'Prediction'))
    plt.tight_layout()
    plt.savefig(name + "_200", dpi=500)
    plt.clf()

    plt.plot(real)
    plt.plot(predicted)
    plt.ylabel("Signal value")
    plt.xlabel("Step")
    plt.xlim(0, 700)
    plt.ylim(-15, 15)
    plt.title('Real versus predicted signal')
    plt.legend(('Real', 'Prediction'))
    plt.tight_layout()
    plt.savefig(name + "_all", dpi=500)
    plt.clf()

    # calculate mean absolute error
    losses = []
    error = 0
    for i in range(0, len(real)):
        error += abs(real[i] - predicted[i])
        losses.append(error/(i+1))

    plt.plot(losses)
    plt.ylabel("Mean absolute error")
    plt.xlabel("M")
    plt.xlim(0, 700)
    plt.title('Mean absolute error as a function of M')
    plt.tight_layout()
    plt.savefig(name + "_loss", dpi=500)
    plt.clf()

    return losses[29], losses[99], losses[199], losses[-1]

def only_losses(file):
    with open(file, "r") as infile:
        reader = csv.reader(infile)
        real = []
        predicted = []
        for row in reader:
            if not row[0] == "Real":
                real.append(float(row[0]))
                predicted.append(float(row[1]))

    # calculate mean absolute error
    losses = []
    error = 0
    for i in range(0, len(real)):
        error += abs(real[i] - predicted[i])
        losses.append(error / (i + 1))

    return losses[29], losses[99], losses[199], losses[-1]

if __name__ == '__main__':

    network = 'Network_LSTM_100_100_100_FC_200_200_gaussian_noise_0,2'

    input_folder = './Additional_frequency_multistep_prediction/' + network + '/'
    output_folder = './Plots/MultiStep/' + network + '/'
    #os.mkdir(output_folder)

    files = get_filepaths(input_folder)

    #for path, n in files:
        #name = output_folder + n.split('.')[0]
        #generate_plots(path, name)

    with open('./Plots/MultiStep/Losses.csv', 'a+', newline='') as outfile:
        writer = csv.writer(outfile)

        for path, n in files:
            loss = only_losses(path)
            row = [network + "-" + n.split('.')[0]]
            row.extend(loss)
            writer.writerow(row)

