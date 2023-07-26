from Helper_functions import getBinRanges, discretize

import csv

# requirement: pandas
import pandas as pd

# TODO: implement MDRM Sensitivity analysis as class and then write a
# "dimension reduction" wrapper function in BNdata - enables to reduce number
# of BN input variables.

class BNdata:
    """
    The BNdata class is designed for reading, storing and binning data for
    Bayesian Network analysis. This class can take a CSV file or a list of
    lists as input, perform binning based on  user-defined parameters, and
    store binned data and related details for further analysis.
    """

    def __init__(self, csvdata, targetlist, binTypeDict, numBinsDict):
    # def __init__(self, csvdata, targetlist, **kwargs):
        """
        Initializes the BNdata class.

        Parameters
        ----------
        csvdata : str or list of lists
            Data to be used, provided either as a file path to a CSV file, or
            a list of lists where each list represents a row of data.
        targetlist : list
            List of target variable names. These variables are of interest in
            the Bayesian Network analysis.
        binTypeDict : dict
            A dictionary mapping variable names to binning types (e.g.,
            ``"equal"``, ``"percentile"``).
        numBinsDict : dict
            A dictionary mapping variable names to the number of bins to use
            for that variable.
        """

        # if "binTypeDict" in kwargs:
        #    self.binTypeDict = kwargs["binTypeDict"]

        self.targets = targetlist
        self.numBinsDict = numBinsDict
        self.binTypeDict = binTypeDict

        print "importing data from csv file ..."

        if isinstance(csvdata, basestring):
            # data is a filepath
            dataset = []
            with open(csvdata, "rb") as csvfile:
                lines = csv.reader(csvfile)

                for row in lines:
                    dataset.append(row)

            csvD = []
            # csvD.append(dataset[0])
            for i in range(0, len(dataset)):
                row = []
                for j in range(0, len(dataset[i])):
                    if i == 0:
                        row.append(dataset[i][j])
                    else:
                        item = float(dataset[i][j])
                        row.append(item)
                csvD.append(row)
            # print np.array(data).astype(np.float)

            self.dataArray = csvD
            self.data = pd.DataFrame(data=csvD[1:], columns=csvD[0])
        elif isinstance(csvdata, list):
            # data is a list of lists
            self.data = pd.DataFrame(csvdata, header=0)

            self.dataArray = csvdata

        print "importing data from csv file completed"

        ##################################################################
        # range discretization using equal or percentile binning
        ##################################################################

        # returns dict with bin ranges
        binRanges = getBinRanges(self.data, self.binTypeDict, self.numBinsDict)
        self.binRanges = binRanges

        ##################################################################
        # range discretization using minimum length description method
        ##################################################################
        # binRanges = getBinRangesAuto(self.data, targetlist)

        # if "numBinsDict" in kwargs:
        #    self.numBinsDict = kwargs["numBinsDict"]
        # else:
        #    self.binRanges = binRanges
        #    self.numBinsDict = {}
        #    for var in binRanges:
        #        self.numBinsDict[var]=len(binRanges[var])

        print "binning data ..."

        datadf, datadict, bincountsdict = discretize(self.data, self.binRanges, True)

        print "binning data complete"

        self.binnedDict, self.binnedData, self.bincountsDict = datadf, datadict, bincountsdict

    """
    def loadFromCSV(self, header=False):
        # TODO: should rewrite this function as loaddataset_kfold and write
        # the kfold code in here and return list of lists of indexes
        dataset = []
        with open(self.file, "rb") as csvfile:
            lines = csv.reader(csvfile)

            for row in lines:
                dataset.append(row)
        data = []
        if (header == True): data.append(dataset[0])
        for i in range(0, len(dataset)):
            row = []
            for j in range(0, len(dataset[i])):
                if i == 0:
                    row.append(dataset[i][j])
                else:
                    item = float(dataset[i][j])
                    row.append(item)
            data.append(row)
        # print np.array(data).astype(np.float)
        self.data = data

        return data


    def getBinRanges(self, binTypeDict, numBinsDict):

        # percentileBoolDict should be in the form of:
        # {
        #     "max_def": False,
        #     "moment_inertia": True,
        #     ...
        # }
        # numBinDict should be in the form of:
        # {
        #     "max_def": 10,
        #     "moment_inertia": 5,
        #     ...
        # }

        # trainingDf = pd.DataFrame(self.data)
        # trainingDf.columns = trainingDf.iloc[0]
        # trainingDf = trainingDf[1:]
        # print trainingDf

        trainingDfDiscterizedRanges = []
        trainingDfDiscterizedRangesDict = {}

        for varName in list(self.data):
            # looping through variables in trainingDf (columns) to discretize
            # into ranges according to trainingDf

            # key = traininDf.columns

            # if true, discretise variable i, using percentiles, if false,
            # discretise using equal bins
            if binTypeDict[varName] == "percentile":
                # add to list
                trainingDfDiscterizedRanges.append(percentile_bins(self.data[varName], numBinsDict.get(varName)))
                # adds to a dict
                trainingDfDiscterizedRangesDict[varName] = percentile_bins(self.data[varName], numBinsDict.get(varName))
            elif "equal":
                # add to list
                trainingDfDiscterizedRanges.append(bins(max(self.data[varName]), min(self.data[varName]), numBinsDict.get(varName)))
                # adds to a dict
                trainingDfDiscterizedRangesDict[varName] = bins(max(self.data[varName]), min(self.data[varName]), numBinsDict.get(varName))

        # update class attribute, while you're at it
        self.bin_ranges = trainingDfDiscterizedRangesDict

        return trainingDfDiscterizedRangesDict


    def discretize(self, binRangesDict, plot=False):
        # percentileBoolDict should be in the form of:
        # {
        #     "max_def": False,
        #     "moment_inertia": True,
        #     ...
        # }
        # numBinDict should be in the form of:
        # {
        #     "max_def": 10,
        #     "moment_inertia": 5,
        #     ...
        # }

        # df = pd.DataFrame(data)
        # df.columns = df.iloc[0]
        # df = df [1:]
        # print df

        binnedDf = pd.DataFrame().reindex_like(self.data)

        # copy trainingDfDiscterizedRangesDict
        binCountsDict = copy.deepcopy(binRangesDict)
        for key in binCountsDict:
            for bin in binCountsDict[key]:
                del bin[:]
                bin.append(0)

        # for tr_row, val_row in itertools.izip_longest(trainingdata, validationdata):
        for varName in list(self.data):
            # load discretized ranges belonging to varName in order to bin in
            discreteRanges = binRangesDict.get(varName)
            # binCounts = binCountsDict[varName]

            index = 0
            # for item1, item2 in trainingDf[varName], valBinnedDf[varName]:
            for item1 in self.data[varName]:
                for i in range(len(discreteRanges)):
                    binRange = discreteRanges[i]

                    ############ bin training data #############

                    if binRange[0] <= item1 <= binRange[1]:
                        # print item1, " lies within ", binRange
                        binnedDf.iloc[index][varName] = i
                        binCountsDict[varName][i][0] += 1

                    if i == 0 and binRange[0] > item1:
                        # print "the value ", item1, "is smaller than the minimum bin", binRange[0]
                        binnedDf.iloc[index][varName] = i
                        binCountsDict[varName][i][0] += 1

                    if i == len(discreteRanges) - 1 and binRange[1] < item1:
                        # print "the value ", item1, "is larger than the maximum bin", binRange[1]
                        binnedDf.iloc[index][varName] = i
                        binCountsDict[varName][i][0] += 1

                index += 1

        binnedData = binnedDf.to_dict(orient="records") # a list of dictionaries
        self.binnedData = binnedData

        print "train binCountdict ", binCountsDict
        print "binned_trainingData ", binnedData

        return binnedData
    """
