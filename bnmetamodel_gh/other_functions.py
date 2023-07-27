from __future__ import print_function

# requirement: sklearn
from sklearn.model_selection import train_test_split

# requirement: pandas
import pandas as pd

# requirement: numpy
import numpy as np

import csv
import copy
import random
import numbers

def numericalSort(value):
    """
    Splits the input string on numbers and sorts the parts numerically.

    Parameters
    ----------
    value : str
        Input string to be sorted.

    Returns
    -------
    parts : list
        The parts of the input string sorted numerically.
    """
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def loadDataset(filename, split, training_data=[], ver_data=[]):
    """
    Load a dataset from a CSV file and split it into training and verification
    sets.

    Parameters
    ----------
    filename : str
        Path to the CSV file containing the dataset.
    split : float
        Fraction of the data to be used for training.
    training_data : list, optional
        Existing list to which training data will be appended. Defaults to an
        empty list.
    ver_data : list, optional
        Existing list to which verification data will be appended. Defaults to
        an empty list.
    """
    with open(filename, "rb") as csvfile:
        data = csvfile.read()
        data = data.decode('utf-8')
        lines = csv.reader(data)
        dataset = list(lines)
    training_data.append(dataset[0])
    ver_data.append(dataset[0])

    for x in range(1, len(dataset) - 1):
        for y in range(len(dataset[x])):
            dataset[x][y] = float(dataset[x][y])
        if random.random() < split:
            training_data.append(dataset[x])
        else:
            ver_data.append(dataset[x])

    print("Xtrain_old", training_data)
    print("X_test)old", ver_data)

def loadDataset_sk(filename, training_data=[], ver_data=[]):
    """
    Load a dataset from a CSV file and split it into training and verification
    sets using the ``sklearn`` library.

    Parameters
    ----------
    filename : str
        Path to the CSV file containing the dataset.
    training_data : list, optional
        Existing list to which training data will be appended. Defaults to an
        empty list.
    ver_data : list, optional
        Existing list to which verification data will be appended. Defaults to
        an empty list.

    Returns
    -------
    tuple : tuple
        Tuple containing training data and verification data.
    """
    with open(filename, "rb") as csvfile:
        data = csvfile.read()
        data = data.decode('utf-8')
        lines = csv.reader(data)
        dataset = list(lines)
    header = dataset[0]
    del dataset[0]

    n_dataset = []

    for i in range(1, len(dataset)):
        n_dataset.append([float(j) for j in dataset[i]])

    training_data, ver_data = train_test_split(n_dataset, test_size=0.33, random_state = None)

    training_data.insert(0, header)
    ver_data.insert(0, header)

    """
    for i in range(1, len(training_data)):
        for j in range(len(training_data[i])):
            float(training_data[i][j])

    for i in range(1, len(ver_data)):
        for j in range(len(ver_data[i])):
            float(ver_data[i][j])
    """
    print("len Xtrain", len(training_data))
    print("len X_test", len(ver_data))
    return training_data, ver_data

def generate_training_ver_data(csv_file_path, num_ver_samples):
    """
    Generates training and verification data sets from a CSV file.

    Parameters
    ----------
    csv_file_path : str
        Path to the CSV file.
    num_ver_samples : int
        Number of samples to include in the verification set.

    Returns
    -------
    tuple : tuple
        Tuple containing training data and verification data.
    """

    # READ CSV DATA

    # data_array = []
    data = []

    with open(csv_file_path, "rb") as f:
        data = f.read()
        data = data.decode('utf-8')
        reader = csv.reader(data, dialect = csv.excel)

        for row in reader:
            data.append(row)

    # SPLIT DATA INTO "TRAINING" DATA AND "VERIFICATION" DATA

    ver_data = []
    training_data = copy.copy(data)

    # rn = np.random.uniform(1, len(training_data), num_ver_samples)
    # print(rn [len(rn)-1])
    # print(num_ver_samples)
    # random_numbers =  []

    # for i in range(0, len(rn)):
    #    random_numbers.append(float(rn[i]))
    #    random_numbers[i] = int(random_numbers[i])

    # print(random_numbers)
    random_numbers = random.sample(range(1, len(training_data)), num_ver_samples)

    ver_data.append(data[0])

    for i in range(0, len(random_numbers)):
        r = random_numbers[i]
        # print("r =", r)
        ver_data.append(training_data[r])
        training_data[r] = 0

    training_data = filter(lambda a: a != 0, training_data)
    # print(training_data)
    # print(ver_data)
    return training_data, ver_data

def list_to_libpgm_dict(list):
    """
    Convert a list to a dictionary compatible with libpgm.

    Parameters
    ----------
    list : list
        List to be converted to a dictionary.

    Returns
    -------
    data_array : list
        List of dictionaries where each dictionary corresponds to a list
        element.
    """
    # print("l", list)
    data_array = []

    for i in range(1, len(list)):
        # print("l1", list[i])
        temp_dict = {}
        # print("i is", i)
        for j in range(0, len(list[i])):
            # print(list[i][j])
            temp_dict[str(list[0][j])] = float(list[i][j])

        data_array.append(temp_dict)

    return data_array

def discretize(data, vars_to_discretize, n_bins):
    """
    Discretize selected variables in the data.

    Parameters
    ----------
    data : dict
        Dictionary containing the dicretization type for selected variables.
    vars_to_discretize : dict
        Dictionary where keys are variable names to be discretized and values
        are discretization types ("Equal", "Freq", or "Bins").
    n_bins : dict
        Dictionary where keys are variable names to be discretized and values
        are the number of bins for each variable.

    Returns
    -------
    tuple : tuple
        Tuple where the first element is the discretized data and the second
        element is a dictionary of bin definitions for each variable.
    """
    data_subset = pd.DataFrame(data).copy()
    bins = {}
    for i in vars_to_discretize:

        out = None
        binning = None

        # discretize by splitting into equal intervals
        if vars_to_discretize[i] == "Equal":
            out, binning = pd.cut(data_subset.ix[:, i], bins = n_bins[i], labels = False, retbins = True)

        # discretize by frequency
        elif vars_to_discretize[i] == "Freq":
            nb = n_bins[i]
            while True:
                try:
                    out, binning = pd.qcut(data_subset.ix[:, i], q = nb, labels = False, retbins = True)
                    break
                except:
                    nb -= 1

        # discretize based on provided bin margins
        elif vars_to_discretize[i] == "Bins":
            out = np.digitize(data_subset.ix[:, i], n_bins[i], right = True) - 1
            binning = n_bins[i]

        data_subset.ix[:, i] = out

        # replace NA variables with and special index (1 + max) -
        # if it has not been done so automatically an in np.digitize
        data_subset.ix[:, i][data_subset.ix[:, i].isnull()] = data_subset.ix[:, i].max() + 1
        bins[i] = binning

    return data_subset, bins

def ranges_extreme(csvData):
    """
    Compute the range (min and max) of each variable in the data.

    Parameters
    ----------
    csvData : list
        List of lists containing the data with each inner list representing a
        variable.

    Returns
    -------
    ranges : dict
        Dictionary where keys are variable names and values are lists with two
        elements [min, max].
    """
    ranges = {}

    data = copy.deepcopy(csvData)
    data = zip(*data)
    # print(data)

    for i in range(0, len(data)):
        var_name = data[i][0]

        data[i] = list(data[i])

        data[i].remove(data[i][0])

        data[i] = map(float, data[i])

        # print("dataaaa", data[i])
        # print("min of this dataaa", min(data[i]))

        ranges[str(var_name)] = [float(min(list(data[i]))), float(max(list(data[i])))]

    return ranges

def valstobins(csvData, val_dict, numBins):
    """
    Convert values to bin indices based on the bin ranges computed from the
    data.

    Parameters
    ----------
    csvData : list
        List of lists containing the data with each inner list representing a
        variable.
    val_dict : dict
        Dictionary where keys are variable names and values are the values to
        be converted to bin indices.
    numBins : int
        Number of bins.

    Returns
    -------
    output_bins : dict
        Dictionary where keys are variable names and values are the bin
        indices.
    """

    # typical val_dict looks like this: {"A": "0.1", ...}
    output_bins = {}

    #    e = {}
    #    for i in range(0, len(val_dict)):
    #        print("eeee", extreme_ranges_dict[val_dict.keys()[i]])
    #        if extreme_ranges_dict[val_dict.keys()[i]] != None:
    #            e[val_dict.keys()[i]] = extreme_ranges_dict[val_dict.keys()[i]]

    # extract ranges of bins from extreme ranges
    extreme_ranges_dict = ranges_extreme(csvData)
    # extreme_ranges_dict  = ranges(data)
    extreme_ranges = list(extreme_ranges_dict)
    # extreme_ranges = list(extreme_ranges_dict.values())
    # print("val dict", val_dict)

    for key in val_dict.keys():
        # for i in range(0, len(val_dict)):
        # min = extreme_ranges_dict[val_dict.keys()[i]][0]
        # max = extreme_ranges_dict[val_dict.keys()[i]][1]

        min = extreme_ranges_dict[key][0]
        max = extreme_ranges_dict[key][1]
        # print("min", min\  \   Q$WZsr)
        # print("max", max)


        bin_ranges = bins(max, min, numBins)
        print("bin range for ", key, bin_ranges)

        for j in range(0, len(bin_ranges)):

            val_check = val_dict[key]
            print("value to check", val_check)
            bin_min = bin_ranges[j][0]
            # print("bin min", bin_min)
            bin_max = bin_ranges[j][1]
            # print("bin max", bin_max)

            # print("val", val_dict.values()[i])

            # if ((val_dict.values()[i] >= bin_min) and (val_dict.values()[i] <= bin_max)) :
            # output_bins[str(val_dict.keys()[i])] = j

            if ((val_check >= bin_min) and (val_check <= bin_max)):
                output_bins[str(key)] = j

    # print(output_bins)
    return output_bins

def whichBin(values_list, ranges_list, indexOnly = False):
    """
    Determine which bin each value in a list falls into based on a list of bin
    ranges.

    Parameters
    ----------
    values_list : list
        List of values to be binned.
    ranges_list : list
        List of bin ranges.
    indexOnly : bool, optional
        If True, return only the bin indices. If False, return a list of lists
        where each inner list is a binary list indicating the bin a value falls
        into. Defaults to False.

    Returns
    -------
    binned_list : list
        List of bin indices or a list of binary lists depending on the value of
        `indexOnly`.
    """
    binned_list = []
    bin_index_list = [0] * len(values_list)

    print("ranges ", ranges_list)

    for i in range(len(values_list)):
        # print("value to bin ", values_list[i])
        binned = []
        for k in range(len(ranges_list)):
            binned.append(0.0)

        # print("--------------[ ", i, " ]---------------")
        # print("range ", ranges_list)
        # print("value to bin", values_list[i])

        for j in range(len(ranges_list)):
            if ((values_list[i] >= ranges_list[j][0]) & (values_list[i] <= ranges_list[j][1])):
                binned[j] = 1.0
                # print("bin found ", j)
                bin_index_list[i] = j
            # elif (j == len(ranges_list)-1) :

        binned_list.append(binned)

    # print("len of bin index list ", len(bin_index_list))
    # print("len of binned list ", len(binned_list))
    print("bin index list", bin_index_list)

    if indexOnly == True :
        return bin_index_list

    return binned_list

def binstovals(bin_ranges):
    """
    This function seems to be incomplete. The purpose and parameters are
    unknown due to lack of code body.

    Parameters
    ----------
    bin_ranges : unknown
        The purpose and type of this parameter is unknown due to lack of
        function body.

    Returns
    -------
    None
    """
    # for i in range(0, bin_ranges):

    # print(output_bins)
    return

def disc2(csv_data, data, alldata, numBins, minmax):
    """
    Discretizes the data based on ranges defined by extreme values, using
    either percentile discretization or equal distance discretization.

    Parameters
    ----------
    csv_data : list of dicts
        Raw data input from a CSV file.
    data : list of dicts
        Data to be discretized.
    alldata : list of dicts
        Additional data used for determining ranges of discretization.
    numBins : int
        Number of bins to discretize the data into.
    minmax : dict
        Dictionary mapping keys in the data to a tuple of their min and max
        values.

    Returns
    -------
    list of dicts
        Discretized data.
    """
    assert (isinstance(data, list) and data and isinstance(data[0], dict)), "Arg must be a list of dicts."
    cdata = copy.deepcopy(data)

    # extreme_ranges_dict  = ranges(cdata)

    extreme_ranges_dict = ranges_extreme(csv_data)

    # print("data", data)
    binned_data = []
    print("csv_data", csv_data)

    df = pd.DataFrame(csv_data)
    df.columns = df.iloc[0]
    df = df[1:]

    print("all data", alldata)

    alldf = pd.DataFrame(alldata)
    alldf.columns = alldf.iloc[0]
    alldf = alldf[1:]

    all_ranges = []
    output_ranges = []
    all_key_strings = df.columns.get_values()
    all_key_strings = all_key_strings.tolist()
    # print(all_key_strings)

    for i in range(len(df.columns)):
        # store key name
        # all_key_strings.append(df.columns[i].astype(str))
        # [
        #   [0.5901, 1.072859], [1.072859, 2.220474], [2.220474, 4.197012],
        #   [4.197012, 6.620893], [6.620893, 9.349943], [9.349943, 13.694827],
        #   [13.694827, 18.286964], [18.286964, 24.310064],
        #   ...
        # ]
        all_ranges.append(percentile_bins(alldf[alldf.columns[i]], numBins))

        if i == 0:
            output_ranges.append(percentile_bins(alldf[alldf.columns[i]], numBins))

        # print(list (col))

    print("all ranges ", all_ranges)

    # for sample in cdata:
    # print(len(cdata))
    for i in range(0, len(cdata)):
        output_bins = {}
        # print("length of ---------------------------", i, "is", len(cdata[i]))
        counter = 0
        for key in cdata[i].keys():
            # for j in range(0, len(cdata[i])):
            # print("j iter is", j)
            # print("cdata key ", key)
            min = minmax [key][0]
            # min = extreme_ranges_dict[key][0]
            # print("min ", min)
            max = minmax [key][1]
            # max = extreme_ranges_dict[key][1]

            index = all_key_strings.index(key)

            # kkk = str(csv_data[0][counter])

            # print("kkk ", key, index, kkk)
            # print("kkk ", kkk)
            # print("max ", max)

            # min = float(extreme_ranges_dict[cdata[i].keys()[j]][0])
            # max = float(extreme_ranges_dict[cdata[i].keys()[j]][1])
            # print("counter = ", counter)
            # TODO: change hard coded max_def to output via constructor
            if key == "max_def":
                # using percentile discretisation
                # bin_ranges = all_ranges[index]
                # bin_ranges = output_ranges[index]

                # using equal distance discretisation
                bin_ranges = bins(max, min, numBins)
                # print("bin ranges disc2 ", bin_ranges)
            else:
                # using percentile discretisation
                bin_ranges = all_ranges[index]

                # using equal distance discretisation
                # bin_ranges = bins(max, min, numBins)

                # print("not max_def")

                # print("old bin ranges ", bin_ranges)

            # bin_ranges = all_ranges[index]
            # print("new bin ranges ", bin_ranges)

            counter = counter + 1

            for k in range(0, len(bin_ranges)):
                val_check = round(cdata[i][key], 6)
                # print("val to be checked = ", val_check)

                bin_min = bin_ranges[k][0]
                bin_max = bin_ranges[k][1]

                # print("bin min", bin_min)
                # print("bin max", bin_max)

                # print("val", val_dict.values()[i])
                if ((val_check >= bin_min) and (val_check <= bin_max)):
                    # print("key", key)
                    # if (output_bins[str(key)] != None):
                    if key not in output_bins:
                        output_bins[str(key)] = k

                if (k == 0) and (val_check<bin_min):
                        output_bins[str(key)] = k

                # if (k == len(bin_ranges)-1) and (val_check > bin_max):
                    # output_bins[str(key)] = k

                    # print(str(cdata[i].keys()[j]))
                    # print("k = ", k)
                # else: print("whooooooooops !")

                # if ((cdata[i].values()[j] >= bin_min) and (cdata[i].values()[j] <= bin_max)) :
                #     output_bins[str(cdata[i].keys()[j])] = k
                #     print(str(cdata[i].keys()[j]))
                #     print(k)

        binned_data.append(output_bins)

    print("binned data", binned_data)
    return binned_data

def disc3(csv_data, data, numBins):
    """
    Discretizes the data based on ranges defined by extreme values, with the
    binned value being the midpoint of the bin.

    Parameters
    ----------
    csv_data : list of dicts
        Raw data input from a CSV file.
    data : list of dicts
        Data to be discretized.
    numBins : int
        Number of bins to discretize the data into.

    Returns
    -------
    list of dicts
        Discretized data with each value being the midpoint of the bin it
        belongs to.
    """
    assert (isinstance(data, list) and data and isinstance(data[0], dict)), "Arg must be a list of dicts."
    cdata = copy.deepcopy(data)

    # extreme_ranges_dict  = ranges(cdata)
    extreme_ranges_dict = ranges_extreme(csv_data)

    # print(extreme_ranges_dict)

    binned_data = []

    # for sample in cdata:
    # print(len(cdata))
    for i in range(0, len(cdata)):
        output_bins = {}
        # print("length of---------------------------", i, "is", len(cdata[i]))

        for key in cdata[i].keys():
            # for j in range(0, len(cdata[i])):
            # print("j iter is", j)
            # print("cdata", key)

            min = extreme_ranges_dict[key][0]
            # print(min)
            max = extreme_ranges_dict[key][1]
            # print(max)

            # min = float(extreme_ranges_dict[cdata[i].keys()[j]][0])
            # max = float(extreme_ranges_dict[cdata[i].keys()[j]][1])

            bin_ranges = bins(max, min, numBins)
            # print(key, bin_ranges)

            for k in range(0, len(bin_ranges)):
                val_check = round(cdata[i][key], 6)
                # print("val to be checked = ", val_check)
                bin_min = bin_ranges[k][0]
                # print("bin min", bin_min)
                bin_max = bin_ranges[k][1]
                # print("bin max", bin_max)

                # print("val", val_dict.values()[i])
                if ((val_check >= bin_min) and (val_check <= bin_max)):
                    output_bins[str(key)] = ((bin_max - bin_min) / 2) + bin_min
                    # print(str(cdata[i].keys()[j]))
                    # print("k = ", k)
                    # else: print("whooooooooops !")

                    #               if ((cdata[i].values()[j] >= bin_min) and (cdata[i].values()[j] <= bin_max)) :
                    #                   output_bins[str(cdata[i].keys()[j])] = k
                    #                   print(str(cdata[i].keys()[j]))
                    #                   print(k)

        binned_data.append(output_bins)

    # print(binned_data)
    return binned_data

def disc(data, bins):
    """
    Discretizes the data based on ranges defined by the min and max values of
    each feature.

    Parameters
    ----------
    data : list of dicts
        Data to be discretized.
    bins : int
        Number of bins to discretize the data into.

    Returns
    -------
    list of dicts
        Discretized data.
    """
    assert (isinstance(data, list) and data and isinstance(data[0], dict)), "Arg must be a list of dicts."
    cdata = copy.deepcopy(data)
    # cdata = data

    # establish ranges
    ranges = dict()
    for variable in cdata[0].keys():
        ranges[variable] = [float("infinity"), float("infinity") * -1]
    for sample in cdata:
        for var in sample.keys():
            if sample[var] < ranges[var][0]:
                ranges[var][0] = sample[var]
            if sample[var] > ranges[var][1]:
                ranges[var][1] = sample[var]

                # print(ranges)

                # discretize cdata set
                # bincounts = dict()
                # for key in cdata[0].keys():

                # populate bincounts with keys and 0 values
                # bincounts[key] = [0 for _ in range(bins)]

    for sample in cdata:
        # for sample in range(37, len(cdata)):
        for i in range(bins):
            # print("-------------------------------bin number", i)
            # print(len(sample.keys()))
            for var in sample.keys():

                # print("ranges [var][0]", ranges[var][0])
                # print("ranges [var][1]", ranges[var][1])
                # print("min", (ranges[var][0] + (ranges[var][1] - ranges[var][0]) * i / float(bins)))
                # print("max", (ranges[var][0] + (ranges[var][1] - ranges[var][0]) * (i + 1) / float(bins)))
                # print("sample[var] before", sample[var])

                if (sample[var] >= (ranges[var][0] + (ranges[var][1] - ranges[var][0]) * i / float(bins)) and (sample[var] <= (ranges[var][0] + (ranges[var][1] - ranges[var][0]) * (i + 1) / float(bins)))):
                    # print(sample[var], "goes in bin number", i)
                    # print("yes")
                    sample[var] = i
                    # print("sample[var] after", sample[var])
                    # bincounts[var][i] += 1
                    # else: print("no")

    # print("cdata", cdata)
    # print("binscount", bincounts)
    return cdata
