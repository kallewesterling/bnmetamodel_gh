from Helper_functions import *

# requirement: sklearn
from sklearn.model_selection import KFold

# requirement: pybbn
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.pptc.inferencecontroller import InferenceController
from pybbn.graph.factory import Factory
from pybbn.graph.jointree import EvidenceType

# requirement: pandas
import pandas as pd

# requirement: matplotlib
import matplotlib.pyplot as plt

import copy
from itertools import product


class BayesianNetwork:
    """
    BayesianNetwork class for managing a Bayesian Network.

    The class provides functionalities for initializing a Bayesian Network with
    the option of loading a pre-built model or creating a new model from
    supplied data. It also offers methods for Bayesian inference and generating
    probabilities based on this network.
    """

    def __init__(self, BNdata=None, netStructure=None, modeldata=None, targetlist=None, binranges=None, priors=None):
        """
        Initialise the BayesianNetwork class.

        Parameters
        ----------
        BNdata : DataFrame or None, optional
            The data for the Bayesian Network, if None a pre-built model must
            be supplied, by default None.
        netStructure : GraphSkeleton or str or None, optional
            The structure of the Bayesian Network which can be a GraphSkeleton
            or a file path of the network, required when BNdata is provided, by
            default None.
        modeldata : dict or None, optional
            The pre-built model data in json format, if None a new model will
            be built based on BNdata and netStructure, by default None.
        targetlist : list or None, optional
            The list of target nodes for the Bayesian Network, by default None.
        binranges : dict or None, optional
            The bin ranges for the Bayesian Network variables, by default None.
        priors : dict or None, optional
            The prior probabilities for the variables in the Bayesian Network,
            by default None.
        """

        ##########################################################
        # CONVENTION: data and binnedData are stored in dataframes
        ##########################################################

        ## CASE: load model from already built BN
        if modeldata != None:
            print "model data has been supplied "
            #### modeldata should be in json dict format ####
            self.json_data = modeldata
            # self.learnedBaynet = DiscreteBayesianNetwork(modeldata)
            self.learnedBaynet = DiscreteBayesianNetwork()
            self.nodes = modeldata["V"]
            self.edges = modeldata ["E"]
            self.Vdata = modeldata ["Vdata"]

            self.targets = targetlist
            # self.numBinsDict = numBinsDict
            self.BinRanges = binranges;

        ## CASE: build new model from data supplied via BNdata and netstructure
        else:
            print "model data has not been supplied"
            # self.binRanges = BNdata.binRanges
            # self.data = BNdata.data
            # self.binnedData = BNdata.binnedData
            self.BNdata = BNdata
            self.structure = netStructure
            self.targets = BNdata.targets

            if isinstance(self.structure, basestring):
                # structure is passed as a file path
                # load file into skeleton
                skel = GraphSkeleton()
                skel.load(self.structure)
                skel.toporder()
                self.skel = skel
            else:
                # structure is passed as loaded graph skeleton
                # given skel
                self.skel = self.structure

            # learn bayesian network
            print "building bayesian network ..."
            # learner = PGMLearner()
            # baynet = learner.discrete_mle_estimateparams(skel, discretized_training_data)
            # baynet = discrete_estimatebn(learner, discretized_training_data, skel, 0.05, 1)

            # using discrete_mle_estimateparams2 written as function in this file, not calling from libpgm
            baynet = discrete_mle_estimateparams2(self.skel, BNdata.binnedDict)

            # TODO: baynet might be redundant since we are building a junction tree.

            print "this is what the libpgm algorithm spits out all data ", self.skel.alldata

            self.learnedBaynet = baynet
            self.nodes = baynet.V
            self.edges = baynet.E
            self.Vdata = baynet.Vdata
            self.json_data = {"V": self.nodes, "E": self.edges, "Vdata": self.Vdata}

            # self.numBinsDict = self.BNdata.numBinsDict
            self.BinRanges = self.BNdata.binRanges

            print "building bayesian network complete"

        print "json data ", self.json_data

        # create BN with pybbn
        bbn = Factory.from_libpgm_discrete_dictionary(self.json_data)

        print "building junction tree ..."
        # create join tree (this must be computed once)
        self.join_tree = InferenceController.apply(bbn)
        print "building junction tree is complete"


    # need to modify to accept skel or skelfile
    def generate(self):
        """
        Generates the Bayesian network from the provided skeleton and
        discretised data.

        Returns
        -------
        baynet : BayesianNetwork
            The generated Bayesian network.
        """

        # learn bayesian network
        # learner = PGMLearner()
        # baynet = learner.discrete_mle_estimateparams(skel, discretized_training_data)
        # baynet = discrete_estimatebn(learner, discretized_training_data, skel, 0.05, 1)

        # using discrete_mle_estimateparams2 written as function in this file, not calling from libpgm
        baynet = discrete_mle_estimateparams2(self.skel, self.binnedData)

        self.nodes = baynet.V
        self.edges = baynet.E
        self.Vdata = baynet.Vdata

        return baynet


    def getpriors(self):
        """
        Retrieves the prior probabilities of the variables in the Bayesian
        network.

        Returns
        -------
        priorPDs : dict
            A dictionary mapping variable names to their respective prior
            probabilities.
        """
        priorPDs = {}

        # min = binRanges[0][0]
        # max = binRanges[len(binRanges)][1]
        # binedges = bins(max, min, len(binRanges))

        bincounts = self.BNdata.bincountsDict

        for varName in bincounts:
            total = sum(sum(x) for x in bincounts[varName])
            priors = []
            for count in bincounts[varName]:
                priors.append(float(count[0]) / float(total))

            priorPDs[varName] = priors

        return priorPDs


    def inferPD(self, query, evidence, plot=False):
        """
        Performs probabilistic inference given a query and evidence.

        Parameters
        ----------
        query : dict
            A dictionary specifying the variables to query.
        evidence : dict
            A dictionary specifying the observed variables.
        plot : bool, optional
            If True, plots the inference results. Default is False.

        Returns
        -------
        queriedMarginalPosteriors : list
            List of marginal posterior probabilities for the queried variables.
        postInferencePDs : dict
            A dictionary mapping variable names to their inferred posterior
            probabilities.
        """
        print "performing inference ..."
        print "building conditional probability table ..."

        fn = TableCPDFactorization(self.learnedBaynet)
        print "conditional probability table is completed"
        print "performing inference with specified hard evidence ..."
        result = condprobve2(fn, query, evidence)

        print "result ofrom condprobve2 ", result
        # result = fn.condprobve(query, evidence)

        queriedMarginalPosteriors = []
        postInferencePDs = {}

        if len(query) > 1:

            probabilities = printdist(result, self.learnedBaynet)
            print "probabilities from printdist2 ", probabilities

            for varName in query.keys():
                marginalPosterior = probabilities.groupby(varName, as_index=False)["probability"].sum()
                marginalPosterior.sort_values([varName], inplace = True)
                queriedMarginalPosteriors.append(marginalPosterior)
                postInferencePDs[varName] = marginalPosterior["probability"].tolist()

        else:
            marginalPosterior = printdist(result, self.learnedBaynet)
            marginalPosterior.sort_values([query.keys()[0]], inplace = True)

            # to make sure probabilities are listed in order of bins, sorted by
            # first queried variable
            queriedMarginalPosteriors.append(marginalPosterior)
            postInferencePDs[query.keys()[0]] = marginalPosterior["probability"].tolist()

        # print "evidence keys ", evidence.keys()
        for varName in evidence.keys():
            e = []
            for i in range(0, len(self.BNdata.binRanges[varName])):
                e.append(0.0)

            # print " evidenceMarginalPriors[varName] ", evidenceMarginalPriors[varName]

            e[evidence[varName]] = 1.0
            postInferencePDs[varName] = e

        # print "priors ", postInferencePDs

        print "inference is complete"
        return queriedMarginalPosteriors, postInferencePDs


    def inferPD_2(self, query, evidence, plot=False):
        """
        Perform inference on the Bayesian network using soft evidence.

        Parameters
        ----------
        query : dict
            A dictionary of variables that are being queried.
        evidence : dict
            A dictionary of evidence variables where keys are variable names
            and values are lists of probabilities corresponding to each state
            of the variable.
        plot : bool, optional
            If True, the function will plot the resulting inference. The
            default is False.

        Returns
        -------
        tuple
            The tuple contains two elements. The first element is a list of
            dataframes, each with probability distribution for each queried
            variable. The second element is a dictionary where keys are
            variable names and values are lists of probabilities for each
            state (post inference probability distributions).
        """

        # evidence is provided in the form of a dict:
        # {
        #     "x1": [0.2, 0.1, 0.4, 0.0, 0.3],
        #     "x2": [1.0, 0.0, 0.0, 0.0, 0.0],
        #     ...
        # }

        for varName in evidence.keys():
            # loop through each evidence variable
            var = varName
            num_states = len(evidence[var])

            allStatesQueriedMarginalPosteriors = []
            for i in range(0, num_states):
                # loop through each state
                e = {var: i}

                print "performing inference ..."
                print "building conditional probability table ..."

                # query is list of variables that are being queried
                fn = TableCPDFactorization(self.learnedBaynet)

                print "conditional probability table is completed"
                print "performing inference with specified soft evidence ..."

                result = condprobve2(fn, query, e)

                queriedMarginalPosteriors = []

                if len(query) > 1:
                    probabilities = printdist(result, self.learnedBaynet)

                    for varName in query.keys():
                        marginalPosterior = probabilities.groupby(varName, as_index = False)["probability"].sum()
                        marginalPosterior.sort_values([varName], inplace = True)

                        # returns a list of dataframes, each with probability
                        # distribution for each queried variable
                        queriedMarginalPosteriors.append(marginalPosterior)
                else:
                    marginalPosterior = printdist(result, self.learnedBaynet)
                    marginalPosterior.sort_values([query.keys()[0]], inplace=True)
                    # to make sure probabilities are listed in order of bins,
                    # sorted by first queried variable
                    queriedMarginalPosteriors.append(marginalPosterior)

                # print "queried marginal posteriors ", queriedMarginalPosteriors

                allStatesQueriedMarginalPosteriors.append(queriedMarginalPosteriors)

        # loop through each state
        # convert to dataframe
        assembledPosteriors = []

        # dummy list of queried PD dicts
        assembledP = allStatesQueriedMarginalPosteriors[0]

        # for index, state in enumerate(allStatesQueriedMarginalPosteriors):

        for varName in evidence.keys():
            # loop through each evidence variable
            evidencePD = evidence[varName]
            postInferencePDs = {}
            assembledPosterior = []

            for i, queryVarName in enumerate(query.keys()):
                # print "query varname ", queryVarName

                # num_states= len(self.BNdata.binRanges[queryVarName])
                num_states = len(allStatesQueriedMarginalPosteriors[0][i]["probability"].tolist())

                for j in range(0, num_states):
                    sum = 0
                    for k in range(0, len(evidencePD)):
                    # for k, state in enumerate(allStatesQueriedMarginalPosteriors):

                        # print allStatesQueriedMarginalPosteriors[k][i]["probability"].tolist()[j]
                        # print "evidence ", evidencePD[k]

                        sum += allStatesQueriedMarginalPosteriors[k][i]["probability"].tolist()[j] * evidencePD[k]

                    assembledP[i].set_value(j, "probability", sum) # data frame
                    assembledPosterior.append(sum) # list

                assembledPosteriors.append(assembledPosterior)
                postInferencePDs.update({queryVarName: assembledP[i]["probability"].tolist()})
                # postInferencePDs[queryVarName] = assembledP[i]["probability"].tolist() # for visualising PDs

        # TODO: here need to update BN PDS and set them as priors for infernece with the next evidence variable

        # for visualising evidence PDs
        for evidenceVarName in evidence.keys():
            postInferencePDs[evidenceVarName] = evidence[evidenceVarName]

        print "inference is complete"

        return assembledP, postInferencePDs

    def inferPD_3(self, query, evidence, plot=False):
        """
        Perform inference on the Bayesian network using soft evidence with the
        use of sequences of all possible combinations of states for each
        evidence variable.

        Parameters
        ----------
        query : dict
            A dictionary of variables that are being queried.
        evidence : dict
            A dictionary of evidence variables where keys are variable names
            and values are lists of probabilities corresponding to each state
            of the variable.
        plot : bool, optional
            If True, the function will plot the resulting inference. The
            default is False.

        Returns
        -------
        tuple
            The tuple contains two elements. The first element is a list of
            dataframes, each with probability distribution for each queried
            variable. The second element is a dictionary where keys are
            variable names and values are lists of probabilities for each
            state (post inference probability distributions).
        """

        # evidence is provided in the form of a dict:
        # {
        #   "x1": [0.2, 0.1, 0.4, 0.0, 0.3],
        #   "x2": [1.0, 0.0, 0.0, 0.0, 0.0],
        #   ...
        # }

        ##############################################################
        # GENERATE SEQUENCE DICTIONARY : ALL POSSIBLE COMBINATIONS OF
        # STATES FROM EACH EVIDENCE VARIABLES
        ##############################################################

        allstates = []
        for ev in evidence.keys():
            states = []
            for j in range(len(evidence[ev])):
                states.append(j)

            allstates.append(states)

        sequence = list(product(*allstates))
        sequenceDict = {}
        for name in evidence.keys():
            sequenceDict[name] = []

        for i in range(0, len(sequence)):
            for j, name in enumerate(evidence.keys()):
                # print "val ", c[i][j]
                sequenceDict[name].append(sequence[i][j])

        ##############################################################
        # PERFORM INFERENCE TO GENERATE QUERIED PDs
        # FOR EACH SEQUENCE OF HARD EVIDENCE (SHAOWEI"S METHOD)
        ##############################################################

        allStatesQueriedMarginalPosteriors = []

        # access list of states

        # combinations = [
        #   [
        #       {var: 0},
        #       {var: 0},
        #       {var: 0},
        #       {var: 0},
        #       {var: 0}
        #   ],
        #       [0, 0, 0, 0, 1],
        #       ...
        #   ]
        # ]
        #
        # combinations = {
        #   var: [0, 1, 2, 3, 4, ..., 1],
        #   var: [0, 1, 2, 3, 4, ..., 1],
        #   ...
        # }

        # For each combination of evidence states

        for i in range(0, len(sequence)):
            e = {}
            for var in evidence.keys():
                e[var] = sequenceDict[var][i] # dictionary

            # query is list of variables that are being queried
            fn = TableCPDFactorization(self.learnedBaynet)
            result = condprobve2(fn, query, e)

            queriedMarginalPosteriors = []

            if len(query) > 1:
                probabilities = printdist(result, self.learnedBaynet)

                for varName in query.keys():
                    marginalPosterior = probabilities.groupby(varName, as_index = False)["probability"].sum()
                    marginalPosterior.sort_values([varName], inplace = True)

                    # returns a list of dataframes, each with probability distribution for each queried variable
                    queriedMarginalPosteriors.append(marginalPosterior)

            else:
                marginalPosterior = printdist(result, self.learnedBaynet)

                marginalPosterior.sort_values([query.keys()[0]], inplace = True)

                # to make sure probabilities are listed in order of bins, sorted by first queried variable
                queriedMarginalPosteriors.append(marginalPosterior)

            allStatesQueriedMarginalPosteriors.append(queriedMarginalPosteriors)

        # loop through each state
        assembledPosteriors = [] # convert to dataframe

        # dummy list of queried PD dicts
        assembledP = allStatesQueriedMarginalPosteriors[0]

        postInferencePDs = {}
        assembledPosterior = []

        for i, queryVarName in enumerate(query.keys()):  # for each queried PD
            # print "query varname ", queryVarName

            # num_states = len(self.BNdata.binRanges[queryVarName])
            num_states = len(allStatesQueriedMarginalPosteriors[0][i]["probability"].tolist())  # queried states

            for j in range(0, num_states): # for each state in each queried PD
                sum = 0

                for k in range(0, len(sequence)):
                    # sequence (0, 0), (0, 1), (0, 2), ....
                    # sequenceDict = {var: [0, 1, 2, 3, 4, ..., 1], var: [0, 1, 2, 3, 4, ..., 1], ...}

                    # holds list of probabilities to be multiplied by the
                    # conditional probability
                    ev = []

                    for var in evidence.keys():
                        index = sequenceDict[var][k] # index of evidence state
                        ev.append(evidence[var][index]) # calling the inputted probabilities by index

                    if all(v == 0 for v in ev):
                        continue

                    ################################
                    # ATTEMPT TO TRY STOPPING LOOP WHEN MULTIPLIER IS ZERO
                    # PROBABILITY TO SAVE SPEED
                    ################################
                    multipliers = []
                    for e in ev:
                        if e != 0.0:
                            multipliers.append(e)
                            # print "non zero e ", e

                    # sum += (allStatesQueriedMarginalPosteriors[k][i]["probability"].tolist()[j] * (reduce(lambda x, y: x * y, multiplier)))
                    ########################

                    sum += (allStatesQueriedMarginalPosteriors[k][i]["probability"].tolist()[j] * (reduce(lambda x, y: x * y, ev)))

                assembledP[i].set_value(j, "probability", sum) # data frame
                assembledPosterior.append(sum) # list

            ## this is a cheating step to order probabilities by index of df ... should be fixed somwehre before. Compare results with pybbn and bayesialab
            for d in assembledP:
                d.sort_index(inplace = True)

            assembledPosteriors.append(assembledPosterior)
            postInferencePDs.update({queryVarName: assembledP[i]["probability"].tolist()})

            # postInferencePDs[queryVarName] = assembledP[i]["probability"].tolist() # for visualising PDs

        # for visualising evidence PDs
        for evidenceVarName in evidence.keys():
            postInferencePDs[evidenceVarName] = evidence[evidenceVarName]

        return assembledP, postInferencePDs

    def inferPD_4(self, query, evidence, plot=False):
        """
        Perform inference on the Bayesian network using soft evidence with
        Shaowei's method, which uses a junction tree for inference.

        Parameters
        ----------
        query : dict
            A dictionary of variables that are being queried.
        evidence : dict
            A dictionary of evidence variables where keys are variable names
            and values are lists of probabilities corresponding to each state
            of the variable.
        plot : bool, optional
            If True, the function will plot the resulting inference. The
            default is False.

        Returns
        -------
        tuple
            The tuple contains two elements. The first element is a list of
            dataframes, each with probability distribution for each queried
            variable. The second element is a dictionary where keys are
            variable names and values are lists of probabilities for each
            state (post inference probability distributions).
        """

        # evidence is provided in the form of a dict:
        # {
        #   "x1": [0.2, 0.1, 0.4, 0.0, 0.3],
        #   "x2": [1.0, 0.0, 0.0, 0.0, 0.0],
        #   ...
        # }

        ##############################################################
        # GENERATE SEQUENCE DICTIONARY : ALL POSSIBLE COMBINATIONS OF
        # STATES FROM EACH EVIDENCE VARIABLES
        ##############################################################

        allstates = []
        for ev in evidence.keys():
            states = []
            for j in range(len(evidence[ev])):
                states.append(j)
            allstates.append(states)

        sequence = list(product(*allstates))

        sequenceDict = {}
        for name in evidence.keys():
            sequenceDict[name] = []

        for i in range(0, len(sequence)):
            for j, name in enumerate(evidence.keys()):
                # print "val ", c[i][j]
                sequenceDict[name].append(sequence[i][j])

        print " _________________________________ sequence dict", sequenceDict

        ##############################################################
        # PERFORM INFERENCE TO GENERATE QUERIED PDs
        # FOR EACH SEQUENCE OF HARD EVIDENCE
        ##############################################################
        allStatesQueriedMarginalPosteriors = []

        # access list of states

        # combinations = [
        #   [
        #       {var:0},
        #       {var:0},
        #       {var:0},
        #       {var:0},
        #       {var:0}
        #   ],
        #       [0, 0, 0, 0, 1],
        #       ......
        #   ]
        # ]
        # combinations = {
        #   var: [0, 1, 2, 3, 4, ..., 1],
        #   var: [0, 1, 2, 3, 4, ..., 1],
        #   ...
        # }

        for i in range(0, len(sequence)):
            # Loop through each combination of evidence states

            e = {}
            for var in evidence.keys():
                e[var] = sequenceDict[var][i] # dictionary

            queriedMarginalPosteriors = self.inferWithJunctionTree(e)

            allStatesQueriedMarginalPosteriors.append(queriedMarginalPosteriors)

        # loop through each state
        assembledPosteriors = [] # convert to dataframe

        # dummy list of queried PD dicts
        assembledP = allStatesQueriedMarginalPosteriors[0]

        # for index, state in enumerate(allStatesQueriedMarginalPosteriors):

        # for varName in evidence.keys():  # for each evidence variable

        postInferencePDs = {}
        assembledPosterior = []

        for i, queryVarName in enumerate(query.keys()):
            # loop through each queried PD
            # num_states= len(self.BNdata.binRanges[queryVarName])
            num_states = len(allStatesQueriedMarginalPosteriors[0][i]["probability"].tolist()) # queried states

            for j in range(0, num_states):
                # loop through each state in each queried PD

                sum = 0
                for k in range(0, len(sequence)):
                    # sequence (0, 0), (0, 1), (0, 2), ...
                    # sequenceDict = {var: [0, 1, 2, 3, 4, ..., 1], var: [0, 1, 2, 3, 4, ..., 1], ...}

                    # ``ev`` holds list of probabilities to be multiplied by
                    # the conditional probability
                    ev = []

                    for var in evidence.keys():
                        # index = index of evidence state
                        index = sequenceDict[var][k]

                        # appending with inputted probabilities by index
                        ev.append(evidence[var][index])

                    sum += (allStatesQueriedMarginalPosteriors[k][i]["probability"].tolist()[j] * (reduce(lambda x, y: x * y, ev)))

                assembledP[i].set_value(j, "probability", sum)  # data frame
                assembledPosterior.append(sum)  # list

            ##########################################
            # TODO: this is a cheating step to order probabilities by index of
            # df ... should be fixed somwehre before. Compare results with
            # pybbn and bayesialab
            for d in assembledP:
                d.sort_index(inplace = True)
            ##########################################

            assembledPosteriors.append(assembledPosterior)

            postInferencePDs[list(assembledP[i])[0]] = assembledP[i]["probability"].tolist()

            # postInferencePDs.update({queryVarName: assembledP[i]["probability"].tolist()})
            # postInferencePDs[queryVarName] = assembledP[i]["probability"].tolist() # for visualising PDs

        # for visualising evidence PDs
        for evidenceVarName in evidence.keys():
            postInferencePDs[evidenceVarName] = evidence[evidenceVarName]

        return assembledP, postInferencePDs


    """
    def plotPDs(self, n_rows, n_cols, maintitle, xlabel, ylabel, displayplt = False, **kwargs):

        # calculate the probability densities for the prior distributions
        binRanges = self.BNdata.binRanges
        priorPDs = {}

        bincounts = self.BNdata.bincountsDict

        for varName in bincounts:
            total = sum(sum(x) for x in bincounts[varName])
            priors = []
            for count in bincounts[varName]:
                # print "total ", total
                # print "count ", count[0]

                priors.append(float(count[0]) / float(total))

            priorPDs[varName] = priors

        print priorPDs

        # for varName in binRanges:
        #     draw_barchartpd(binRanges[varName], priorPDs[varName])

        # plot each axes in a figure
        fig = plt.figure(figsize=((200 * n_cols) / 96, (200 * n_rows) / 96), dpi=96)
        t = fig.suptitle(maintitle, fontsize=8)

        i = 0
        for varName in binRanges:
            # print df
            ax = fig.add_subplot(n_rows, n_cols, i + 1)

            xticksv = []
            binwidths = []
            edge = []

            for index, range in enumerate(binRanges[varName]):
                print "range ", range
                edge.append(range[0])
                binwidths.append(range[1] - range[0])
                xticksv.append(((range[1] - range[0]) / 2) + range[0])
                if index == len(binRanges[varName]) - 1: edge.append(range[1])

            # df[var_name].hist(bins = binwidths[var_name], ax=ax)
            ax.bar(xticksv, priorPDs[varName], align="center", width = binwidths, color="black", alpha=0.2)

            evidenceVars = []
            if "evidence" in kwargs:
                evidenceVars = kwargs["evidence"]

            if "posteriorPD" in kwargs:
                if len(kwargs["posteriorPD"][varName]) > 1:
                    print "name ", varName
                    print "hello ", kwargs["posteriorPD"][varName]
                    print "binwidths ", binwidths
                    if varName in evidenceVars:
                        ax.bar(
                            xticksv,
                            kwargs["posteriorPD"][varName],
                            align="center",
                            width=binwidths,
                            color="green",
                            alpha=0.2)

                    else:
                        ax.bar(
                            xticksv,
                            kwargs["posteriorPD"][varName],
                            align="center",
                            width=binwidths,
                            color="red",
                            alpha=0.2)

            # TODO: fix xticks .... not plotting all
            # plt.xlim(edge[0], max(edge))
            plt.xticks([round(e, 4) for e in edge], rotation="vertical")
            plt.ylim(0, 1)
            # plt.show()

            ax.grid(color="0.2", linestyle=":", linewidth=0.1, dash_capstyle="round")
            # ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            ax.set_title(varName, fontweight="bold", size=6)
            ax.set_ylabel(ylabel, fontsize=7)  # Y label
            ax.set_xlabel(xlabel, fontsize=7)  # X label
            ax.xaxis.set_tick_params(labelsize=6)
            ax.yaxis.set_tick_params(labelsize=6)

            # ax.grid(False)
            # if "xlim" in kwargs:
            #    ax.set_xlim(kwargs["xlim"][0], kwargs["xlim"][1])

            i += 1

        # Improve appearance a bit.
        fig.tight_layout()

        # Add white spacing between plots and title
        fig.subplots_adjust(top=0.85)

        # if you want to set background of figure to transparent, do it here.
        # Use facecolor="none" as argument in savefig()

        if displayplt == True:
            plt.show()
    """

    def plotPDs(self, maintitle, xlabel, ylabel, displayplt = False, **kwargs):
        """
        Plots the probability distributions of the nodes in the Bayesian
        Network.

        Parameters
        ----------
        maintitle : str
            The main title of the plot.
        xlabel : str
            The label for the x-axis.
        ylabel : str
            The label for the y-axis.
        displayplt : bool, optional
            If True, the plot is displayed. Default is False.
        **kwargs : dict, optional
            Keyword arguments which can contain evidence variables and
            posterior probability distributions.
        """

        # code to automatically set the number of columns and rows and
        # dimensions of the figure
        n_totalplots = len(self.nodes)
        print "num of total plots ", n_totalplots

        if n_totalplots <= 4:
            n_cols = n_totalplots
            n_rows = 1
        else:
            n_cols = 4
            n_rows = n_totalplots % 4
            print "num rows ", n_rows

        if n_rows == 0:
            n_rows = n_totalplots / 4

        # generate the probability distributions for the prior distributions
        binRanges = self.BNdata.binRanges
        priorPDs = {}

        # min = binRanges[0][0]
        # max = binRanges[len(binRanges)][1]
        # binedges = bins(max, min, len(binRanges))

        bincounts = self.BNdata.bincountsDict
        for varName in bincounts:
            total = sum(sum(x) for x in bincounts[varName])
            priors = []
            for count in bincounts[varName]:
                priors.append(float(count[0]) / float(total))

            priorPDs[varName] = priors

        # for varName in binRanges:
        #     draw_barchartpd(binRanges[varName], priorPDs[varName])

        # instantiate a figure as a placeholder for each distribution (axes)
        width = (200 * n_cols) / 96
        height = (200 * n_rows) / 96
        fig = plt.figure(figsize=(width, height), dpi=96, facecolor="white")

        # Set title
        fig.suptitle(maintitle, fontsize=8)

        # sort evidence distributions to be plotted first
        # nodessorted = []

        # copy node names into new list
        nodessorted = copy.copy(self.nodes)

        # evidence
        evidenceVars = []
        if "evidence" in kwargs:
            evidenceVars = kwargs["evidence"]

            # sort evidence variables to be in the beginning of the list
            for index, var in enumerate(evidenceVars):
                nodessorted.insert(index, nodessorted.pop(nodessorted.index(evidenceVars[index])))

        i = 0
        for varName in nodessorted:
            # print df
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            ax.set_axis_bgcolor("whitesmoke")

            xticksv = []
            binwidths = []
            edge = []

            for index, range in enumerate(binRanges[varName]):
                edge.append(range[0])
                binwidths.append(range[1] - range[0])
                xticksv.append(((range[1] - range[0]) / 2) + range[0])

                if index == len(binRanges[varName]) - 1:
                    edge.append(range[1])

            # df[var_name].hist(bins = binwidths[var_name], ax=ax)
            # plot the priors
            ax.bar(
                xticksv,
                priorPDs[varName],
                align="center",
                width=binwidths,
                color="black",
                alpha=0.2,
                linewidth=0.2)

            # filter out evidence and query to color the bars accordingly (evidence-green, query-red)
            if "posteriorPD" in kwargs:
                if len(kwargs["posteriorPD"][varName]) > 1:
                    if varName in evidenceVars:
                        ax.bar(
                            xticksv,
                            kwargs["posteriorPD"][varName],
                            align="center",
                            width=binwidths,
                            color="green",
                            alpha=0.2,
                            linewidth=0.2)
                    else:
                        ax.bar(
                            xticksv,
                            kwargs["posteriorPD"][varName],
                            align="center",
                            width=binwidths,
                            color="red",
                            alpha=0.2,
                            linewidth=0.2)

            # TODO: fix xticks .... not plotting all
            # plt.xlim(edge[0], max(edge))
            ticks = [round(e, 4) for e in edge]
            plt.xticks(ticks, rotation="vertical")
            plt.ylim(0, 1)
            # plt.show()

            for spine in ax.spines:
                ax.spines[spine].set_linewidth(0)

            ax.grid(color="0.2", linestyle=":", linewidth=0.1, dash_capstyle="round")
            # ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            ax.set_title(varName, fontweight="bold", size=6)
            # Set X and Y label
            ax.set_ylabel(ylabel, fontsize=7)
            ax.set_xlabel(xlabel, fontsize=7)

            ax.xaxis.set_tick_params(labelsize=6, length=0)
            ax.yaxis.set_tick_params(labelsize=6, length=0)

            # ax.grid(False)
            # if "xlim" in kwargs:
            #    ax.set_xlim(kwargs["xlim"][0], kwargs["xlim"][1])

            i += 1

        # Improve appearance a bit.
        fig.tight_layout()

        # Add white spacing between plots and title
        fig.subplots_adjust(top=0.85)

        # if you want to set background of figure to transparent, do it here.
        # Use facecolor="none" as argument in savefig()

        if displayplt == True:
            plt.show()


    def crossValidate(self, targetList, numFolds):
        """
        Performs k-fold cross-validation on the Bayesian Network, returning a
        dictionary of error dataframes, one for each target.

        Parameters
        ----------
        targetList : list
            The list of target variables for which to perform cross-validation.
        numFolds : int
            The number of folds to use for cross-validation.

        Returns
        -------
        dict
            A dictionary where the keys are the targets, and the values are
            pandas dataframes containing the error metrics for each fold of
            the cross-validation.
        """

        # perhaps use **kwargs, to ask if data not specified, then use self.binnedData

        # create empty dataframes to store errors for each target
        error_dict = {}
        for target in targetList:
            df_columns = ["NRMSE", "LogLoss", "Classification Error", "Distance Error"]
            df_indices = ["Fold_%s" % (num + 1) for num in range(numFolds)]
            error_df = pd.DataFrame(index = df_indices, columns = df_columns)
            error_df = error_df.fillna(0.0)
            error_df["Distance Error"] = error_df["Distance Error"].astype(object)
            error_df["Classification Error"] = error_df["Classification Error"].astype(object)

            error_dict[target] = error_df

        # specify number of k-folds
        kf = KFold(n_splits = numFolds)
        kf.get_n_splits((self.BNdata.dataArray))

        fold_counter = 0
        listRMSE = 0.0
        listLogLoss = 0.0
        for training_index, testing_index in kf.split(self.BNdata.data):
            # loop through all data and split into training and testing for each fold
            print "------- FOLD NUMBER ", fold_counter+1, "  ----------------"

            trainingData = kfoldToDF(training_index, self.BNdata.data)
            testingData = kfoldToDF(testing_index, self.BNdata.data)

            # bin test/train data
            binRanges = self.BinRanges
            # binRanges = getBinRanges(trainingData, self.BNdata.binTypeDict, self.BNdata.numBinsDict)
            binnedTrainingDict, binnedTrainingData, binCountsTr = discretize(trainingData, binRanges, False)
            binnedTestingDict, binnedTestingData, binCountsTest = discretize(testingData, binRanges, False)
            binnedTestingData = binnedTestingData.astype(int)

            # estimate BN parameters
            baynet = discrete_mle_estimateparams2(self.skel, binnedTrainingDict)

            queries = {}
            marginalTargetPosteriorsDict = {}
            for target in targetList:
                # assign bin to zero to query distribution (libpgm convention)
                queries[target] = 0
                # create empty list for each target to populate with predicted target posterior distributions
                marginalTargetPosteriorsDict[target] = []

            # In this loop we predict the posterior distributions for each queried target
            # TODO: need to adapt this loop for storing predicted posteriors
            # for each target in the list, and eventually calc error_df for
            # each target (or into one DF with multiple indices)

            for i in range(0, binnedTestingData.shape[0]):
                row = binnedTestingDict[i]
                evidence = without_keys(row, queries.keys())
                fn = TableCPDFactorization(baynet)
                result = condprobve2(fn, queries, evidence)

                if len(queries) > 1:
                    # more than 1 target was specified
                    posteriors = printdist(result, baynet)
                    for target in targetList:
                        marginalPosterior = posteriors.groupby(target)["probability"].sum()
                        # marginalTargetPosteriorsDf[target][i] = marginalPosterior
                        marginalTargetPosteriorsDict[target].append(marginalPosterior) # might need [probability]

                else:
                    # only 1 target was specified
                    posterior = printdist(result, baynet)

                    # to make sure probabilities are listed in order of bins, sorted by first queried variable
                    posterior.sort_values([targetList[0]], inplace=True)

                    # marginalTargetPosteriorsDf[targetList[0]][i] = posterior["probability"]
                    marginalTargetPosteriorsDict[target].append(posterior["probability"])

            # generate accuracy measures at one go
            # for each target
            for key in error_dict.keys():
                rmse, loglossfunction, norm_distance_errors, correct_bin_probabilities = generateErrors(marginalTargetPosteriorsDict[key], testingData, binnedTestingData, binRanges, key)

                # add generated measures to error_df (error dataframe)
                error_dict[key]["NRMSE"][fold_counter] = rmse
                error_dict[key]["LogLoss"][fold_counter] = loglossfunction
                error_dict[key]["Distance Error"][fold_counter] = norm_distance_errors
                error_dict[key]["Classification Error"][fold_counter] = correct_bin_probabilities

            fold_counter += 1
            # listRMSE += rmse
            # listLogLoss += loglossfunction

        # averageRMSE = sum(listRMSE) / float(numFolds)
        # averageLogLoss = sum(listLogLoss) / float(numFolds)

        return error_dict


    def crossValidate_JT(self, targetList, numFolds):
        """
        Performs k-fold cross-validation on the Bayesian Network using junction
        trees, returning a dictionary of error dataframes, one for each target.

        Parameters
        ----------
        targetList : list
            The list of target variables for which to perform cross-validation.
        numFolds : int
            The number of folds to use for cross-validation.

        Returns
        -------
        dict
            A dictionary where the keys are the targets, and the values are
            pandas dataframes containing the error metrics for each fold of the
            cross-validation.
        """

        # perhaps use **kwargs, to ask if data not specified, then use self.binnedData

        # create empty dataframes to store errors for each target
        error_dict = {}
        for target in targetList:
            df_columns = ["NRMSE", "LogLoss", "Classification Error", "Distance Error"]
            df_indices = ["Fold_%s" % (num + 1) for num in range(numFolds)]
            error_df = pd.DataFrame(index = df_indices, columns = df_columns)
            error_df = error_df.fillna(0.0)
            error_df["Distance Error"] = error_df["Distance Error"].astype(object)
            error_df["Classification Error"] = error_df["Classification Error"].astype(object)

            error_dict[target] = error_df

        # specify number of k folds
        kf = KFold(n_splits = numFolds)
        kf.get_n_splits((self.BNdata.dataArray))

        fold_counter = 0
        listRMSE = 0.0
        listLogLoss = 0.0

        # loop through all data and split into training and testing for each fold
        for training_index, testing_index in kf.split(self.BNdata.data):
            print "------- FOLD NUMBER ", fold_counter+1, "  ----------------"

            trainingData = kfoldToDF(training_index, self.BNdata.data)
            testingData = kfoldToDF(testing_index, self.BNdata.data)

            # bin test/train data
            binRanges = self.BinRanges
            # binRanges = getBinRanges(trainingData, self.BNdata.binTypeDict, self.BNdata.numBinsDict)
            binnedTrainingDict, binnedTrainingData, binCountsTr = discretize(trainingData, binRanges, False)
            binnedTestingDict, binnedTestingData, binCountsTest = discretize(testingData, binRanges, False)
            binnedTestingData = binnedTestingData.astype(int)

            # estimate BN parameters
            baynet = discrete_mle_estimateparams2(self.skel, binnedTrainingDict)

            ###########################################
            # JOIN TREE USING PYBBN
            ###########################################
            # get topology of bn
            json_data = {"V": baynet.V, "E": baynet.E, "Vdata": baynet.Vdata}
            # create BN with pybbn
            pybbn = Factory.from_libpgm_discrete_dictionary(self.json_data)
            # create join tree (this must be computed once)
            jt = InferenceController.apply(pybbn)
            ###########################################

            queries = {}
            marginalTargetPosteriorsDict = {}
            for target in targetList:
                # assign bin to zero to query distribution (libpgm convention)
                queries[target] = 0
                # create empty list for each target to populate with predicted target posterior distributions
                marginalTargetPosteriorsDict[target] = []

            # TODO: need to adapt the loop below for storing predicted
            # posteriors for each target in the list, and eventually calc
            # error_df for each Target (or into one DF with multiple indices)

            for i in range(0, binnedTestingData.shape[0]):
                # In this loop we predict the posterior distributions for each
                # queried target

                row = binnedTestingDict[i]
                evidence = without_keys(row, queries.keys())
                # fn = TableCPDFactorization(baynet)

                result = self.inferPD_JT_hard(evidence)
                # result = condprobve2(fn, queries, evidence)

                if len(queries) > 1:
                    # more than 1 target was specified
                    posteriors = printdist(result, baynet)
                    for target in targetList:
                        marginalPosterior = posteriors.groupby(target)["probability"].sum()
                        # marginalTargetPosteriorsDf[target][i] = marginalPosterior
                        marginalTargetPosteriorsDict[target].append(marginalPosterior)  # might need [probability]
                else:
                    # only 1 target was specified
                    posterior = printdist(result, baynet)

                    # to make sure probabilities are listed in order of bins, sorted by first queried variable
                    posterior.sort_values([targetList[0]],
                                          inplace = True)
                    # marginalTargetPosteriorsDf[targetList[0]][i] = posterior["probability"]
                    marginalTargetPosteriorsDict[target].append(posterior["probability"])

            # generate accuracy measures at one go
            for key in error_dict.keys():
                # loop through each target
                rmse, loglossfunction, norm_distance_errors, correct_bin_probabilities = generateErrors(
                    marginalTargetPosteriorsDict[key], testingData, binnedTestingData, binRanges, key)

                # add generated measures to error_df (error dataframe)
                error_dict[key]["NRMSE"][fold_counter] = rmse
                error_dict[key]["LogLoss"][fold_counter] = loglossfunction
                error_dict[key]["Distance Error"][fold_counter] = norm_distance_errors
                error_dict[key]["Classification Error"][fold_counter] = correct_bin_probabilities

            fold_counter += 1
            # listRMSE += rmse
            # listLogLoss += loglossfunction

        # averageRMSE = sum(listRMSE) / float(numFolds)
        # averageLogLoss = sum(listLogLoss) / float(numFolds)

        return error_dict


    def validateNew(self, newBNData, targetList):
        """
        Validates a new Bayesian Network dataset and returns a dictionary of
        error dataframes for each target variable.

        Parameters
        ----------
        newBNData : DiscreteBN
            A new Bayesian Network dataset.
        targetList : list of str
            List of target variables for validation.

        Returns
        -------
        dict
            A dictionary where keys are target variables and values are pandas
            DataFrames containing various error metrics.
        """

        # perhaps use **kwargs, to ask if data not specified, then use self.binnedData

        # create empty dataframes to store errors for each target
        error_dict = {}
        for target in targetList:
            df_columns = ["NRMSE", "LogLoss", "Classification Error", "Distance Error"]
            # df_indices = ["Fold_%s" % (num + 1) for num in range(numFolds)
            df_indices = [0]
            error_df = pd.DataFrame(index = df_indices, columns = df_columns)
            error_df = error_df.fillna(0.0)
            error_df["Distance Error"] = error_df["Distance Error"].astype(object)
            error_df["Classification Error"] = error_df["Classification Error"].astype(object)

            error_dict[target] = error_df

        listRMSE = 0.0
        listLogLoss = 0.0

        # trainingData = kfoldToDF(training_index, self.BNdata.data)
        # testingData = kfoldToDF(testing_index, self.BNdata.data)
        trainingData = self.BNdata.data
        testingData = newBNData.data

        # bin test/train data
        binRanges = self.BinRanges
        # binRanges = getBinRanges(trainingData, self.BNdata.binTypeDict, self.BNdata.numBinsDict)
        binnedTrainingDict, binnedTrainingData, binCountsTr = discretize(trainingData, binRanges, False)
        binnedTestingDict, binnedTestingData, binCountsTest = discretize(testingData, binRanges, False)
        binnedTestingData = binnedTestingData.astype(int)

        # estimate BN parameters
        baynet = discrete_mle_estimateparams2(self.skel, binnedTrainingDict)

        queries = {}
        marginalTargetPosteriorsDict = {}
        for target in targetList:
            # assign bin to zero to query distribution (libpgm convention)
            queries[target] = 0

            # create empty list for each target to populate with predicted
            # target posterior distributions
            marginalTargetPosteriorsDict[target] = []

        # TODO: need to adapt the loop below for storing predicted posteriors for
        # each target in the list, and eventually calc error_df for each target
        # (or into one DF with multiple indices)

        for i in range(0, binnedTestingData.shape[0]):
            # In this loop, we predict the posterior distributions for each
            # queried target

            row = binnedTestingDict[i]
            evidence = without_keys(row, queries.keys())
            fn = TableCPDFactorization(baynet)
            result = condprobve2(fn, queries, evidence)

            if len(queries) > 1:
                # more than 1 target was specified
                posteriors = printdist(result, baynet)
                for target in targetList:
                    marginalPosterior = posteriors.groupby(target)["probability"].sum()
                    # marginalTargetPosteriorsDf[target][i] = marginalPosterior
                    marginalTargetPosteriorsDict[target].append(marginalPosterior) # might need [probability]
            else:
                # only 1 target was specified

                posterior = printdist(result, baynet)

                # to make sure probabilities are listed in order of bins, sorted by first queried variable
                posterior.sort_values([targetList[0]], inplace=True)

                # marginalTargetPosteriorsDf[targetList[0]][i] = posterior["probability"]
                marginalTargetPosteriorsDict[target].append(posterior["probability"])

        # generate accuracy measures at one go
        for key in error_dict.keys():
            # loop through each target
            rmse, loglossfunction, norm_distance_errors, correct_bin_probabilities = generateErrors(marginalTargetPosteriorsDict[key], testingData, binnedTestingData, binRanges, key)

            # add generated measures to error_df (error dataframe)
            error_dict[key]["NRMSE"][0] = rmse
            error_dict[key]["LogLoss"][0] = loglossfunction
            error_dict[key]["Distance Error"][0] = norm_distance_errors
            error_dict[key]["Classification Error"][0] = correct_bin_probabilities

        # listRMSE += rmse
        # listLogLoss += loglossfunction

        # averageRMSE = sum(listRMSE) / float(numFolds)
        # averageLogLoss = sum(listLogLoss) / float(numFolds)

        return error_dict

    # method to perform inference with hard evidence using join tree
    def inferPD_JT_hard(self, hardEvidence):
        """
        Performs inference on the Bayesian Network using hard evidence and the
        junction tree algorithm.

        Parameters
        ----------
        hardEvidence : dict
            A dictionary where keys are variable names and values are the
            observed values (evidence) of those variables.

        Returns
        -------
        dict
            A dictionary where keys are variable names and values are lists
            representing the posterior probabilities of each bin of the
            variables.
        """

        # hardEvidence is supplied in the form:
        # {
        #   "max_def": 5,
        #   "span": 4
        # }
        # converts libpgm to pybnn then use pybnn to run junction tree and then spitback out results for visualising

        print "performing inference using junction tree algorithm ..."

        # convert soft evidence to hard
        formattedEvidence = {}
        for var in hardEvidence.keys():
            for i in range(0, len(hardEvidence[var])):
                if hardEvidence[var][i] == 1.0:
                    formattedEvidence[var] = i

        print "formatted evidence ", formattedEvidence

        def potential_to_df(p):
            """
            Convert the potential of a node into a pandas dataframe.

            Parameters
            ----------
            p : Potential object
                The potential object containing entries.

            Returns
            -------
            df : DataFrame
                A dataframe with columns "val" and "p" representing value and
                probability respectively.
            """
            data = []
            for pe in p.entries:
                v = pe.entries.values()[0]
                p = pe.value
                t = (v, p)
                data.append(t)

            return pd.DataFrame(data, columns=["val", "p"])

        def potentials_to_dfs(join_tree):
            """
            Convert the potentials of all nodes in a join tree into dataframes.

            Parameters
            ----------
            join_tree : JoinTree object
                The join tree whose node potentials are to be converted.

            Returns
            -------
            data : list
                A list of tuples, each containing a variable name and a dataframe
                representing its potential.
            """
            data = []
            for node in join_tree.get_bbn_nodes():
                name = node.variable.name
                df = potential_to_df(join_tree.get_bbn_potential(node))
                t = (name, df)
                data.append(t)

            return data

        def pybbnToLibpgm_posteriors(pybbnPosteriors):
            """
            Convert posteriors from pybbn format to libpgm format.

            Parameters
            ----------
            pybbnPosteriors : list
                A list of tuples, each containing a variable name and a
                dataframe representing its potential.

            Returns
            -------
            posteriors : dict
                A dictionary where keys are variable names and values are lists
                representing the posterior probabilities of each bin of the
                variables.
            """
            posteriors = {}
            for node in pybbnPosteriors:
                var = node[0]
                df = node[1]
                p = df.sort_values(by=["val"])
                posteriors[var] = p["p"].tolist()

            # a dictionary of lists
            return posteriors

        # generate list of pybnn evidence
        evidenceList = []
        for e in formattedEvidence.keys():
            ev = EvidenceBuilder() \
                .with_node(self.join_tree.get_bbn_node_by_name(e)) \
                .with_evidence(formattedEvidence[e], 1.0) \
                .build()

            evidenceList.append(ev)

        self.join_tree.unobserve_all()
        self.join_tree.update_evidences(evidenceList)

        posteriors = potentials_to_dfs(self.join_tree)

        # print "posteriors from joint tree ", posteriors

        # join tree algorithm seems to eliminate bins whose posterior
        # probabilities are zero. check for missing bins and add them back

        for posterior in posteriors:
            # numbins = self.BNdata.numBinsDict[posterior[0]] # try to reduce dependency on external objects BNdata
            # numbins = self.numBinsDict[posterior[0]]
            numbins = len(self.BinRanges[posterior[0]])

            for i in range(0, numbins):
                if float(i) not in posterior[1]["val"].tolist():
                    # print "bin number ", float(i), " was missing "
                    posterior[1].loc[len(posterior[1])] = [float(i), 0.0]
                    continue

        # TODO: remove the following lines? function should return evidence +
        # query posteriors or only latter? --> decide based on plotting req
        # for i, pos in enumerate(posteriors):
        #    if str(pos[0]) in hardEvidence.keys():
        #        posteriors.remove(pos)
                # del posteriors[i]

        posteriorsDict = pybbnToLibpgm_posteriors(posteriors)
        print "inference is complete ... posterior distributions were generated successfully"

        return posteriorsDict

    def inferPD_JT_soft(self, softEvidence):
        """
        Performs inference on the Bayesian Network using soft evidence
        (virtual) and the junction tree algorithm.

        Parameters
        ----------
        softEvidence : dict
            A dictionary where keys are variable names and values are lists
            representing the likelihood of each bin of the variables.

        Returns
        -------
        dict
            A dictionary where keys are variable names and values are lists
            representing the posterior probabilities of each bin of the
            variables.
        """

        # TODO: currently you can only enter likelihoods. Need to find way to
        # enter probabilities and convert them to likelihoods.

        print "performing inference using junction tree algorithm ..."

        # softEvidence = self.convertEvidence(humanEvidence)

        # print "soft evidence ", softEvidence

        def potential_to_df(p):
            """
            Convert the potential of a node into a pandas dataframe.

            Parameters
            ----------
            p : Potential object
                The potential object containing entries.

            Returns
            -------
            df : DataFrame
                A dataframe with columns "val" and "p" representing value and
                probability respectively.
            """
            data = []
            for pe in p.entries:
                v = pe.entries.values()[0]
                p = pe.value
                t = (v, p)
                data.append(t)

            return pd.DataFrame(data, columns=["val", "p"])

        def potentials_to_dfs(join_tree):
            """
            Convert the potentials of all nodes in a join tree into dataframes.

            Parameters
            ----------
            join_tree : JoinTree object
                The join tree whose node potentials are to be converted.

            Returns
            -------
            data : list
                A list of tuples, each containing a variable name and a dataframe
                representing its potential.
            """
            data = []
            for node in join_tree.get_bbn_nodes():
                name = node.variable.name
                df = potential_to_df(join_tree.get_bbn_potential(node))
                t = (name, df)

                data.append(t)
            return data

        def pybbnToLibpgm_posteriors(pybbnPosteriors):
            """
            Convert posteriors from pybbn format to libpgm format.

            Parameters
            ----------
            pybbnPosteriors : list
                A list of tuples, each containing a variable name and a
                dataframe representing its potential.

            Returns
            -------
            posteriors : dict
                A dictionary where keys are variable names and values are lists
                representing the posterior probabilities of each bin of the
                variables.
            """
            posteriors = {}

            for node in pybbnPosteriors:
                var = node[0]
                df = node[1]
                p = df.sort_values(by=["val"])
                posteriors[var] = p["p"].tolist()

            return posteriors # returns a dictionary of dataframes

        evidenceList = []

        for evName in softEvidence.keys():

            ev = EvidenceBuilder().with_node(self.join_tree.get_bbn_node_by_name(evName))

            for state, likelihood in enumerate(softEvidence[evName]):
                ev.values[state] = likelihood

            # specify evidence type as virtual (soft) (likelihoods, not
            # probabilities)
            ev = ev.with_type(EvidenceType.VIRTUAL).build()
            evidenceList.append(ev)

        self.join_tree.unobserve_all()
        self.join_tree.update_evidences(evidenceList)


        posteriors = potentials_to_dfs(self.join_tree) # contains posteriors + evidence distributions


        # ``posteriordistributions`` contains posterior distributions of
        # queried variables only
        # posteriordistributions = []
        # for i, pos in enumerate(alldistributions):
            # if str(pos[0]) not in softEvidence.keys():
                # posteriordistributions.append(pos)

        # join tree algorithm seems to eliminate bins whose posterior
        # probabilities are zero. the following checks for missing bins and
        # adds them back:

        for posterior in posteriors:
            print "posssssssterior ", posterior
            # numbins = self.BNdata.numBinsDict[posterior[0]]
            numbins = len(self.BinRanges[posterior[0]])

            for i in range(0, numbins):
                if float(i) not in posterior[1]["val"].tolist():
                    # print "bin number ", float(i), " was missing "
                    posterior[1].loc[len(posterior[1])] = [float(i), 0.0]
                    continue


        posteriorsDict = pybbnToLibpgm_posteriors(posteriors)

        """
        posteriordistributionsDict = {}
        for dataf in libpgmPosteriors:
            varname = list(dataf)[0]
            posteriordistributionsDict[varname] = dataf["probabilities"].tolist()

        alldistributionsDict = {}
        for dataf in libpgmAlldistributions:
            varname = list(dataf)[0]
            alldistributionsDict[varname] = dataf["probabilities"].tolist()
        """
        # return posteriordistributionsDict, alldistributionsDict

        print "inference is complete ... posterior distributions were generated successfully"

        return posteriorsDict  # posteriors + evidence distributions (for visualising)

    def convertEvidence(self, humanEvidence):
        """
        Converts human-readable evidence into a format understandable by the
        Bayesian Network.

        This function takes in human evidence, which can either be ranges of
        interest or specific hard numbers, and converts it into a dictionary
        where each key is a variable and each value is a list of probabilities.

        Parameters
        ----------
        humanEvidence : dict
            The human evidence to convert. It should be in the form:
                {v1: [min, max], v2: [min, max]}
            or
                {v1: [val], v2: [val]}
            where v1, v2, etc. are variable names, and the lists contain
            either a range of interest or a specific value for the variable.

        Returns
        -------
        dict
            The converted evidence, in the form
            ``{v1:[0.0, 1.0, 0.2], v2:[0.1, 0.5, 1.0], ...}``, where each key
            is a variable name, and each value is a list of probabilities
            associated with each bin of that variable. The probability is 1.0
            if the bin is within the input range or equals the hard number,
            and 0.0 otherwise.
        """

        # humanEvidence can either be entered as ranges of interest {v1: [min, max], v2: [min, max]} or hard numbers {v1: [val], v2: [val]}
        # need to return a dict {v1:[0.0, 1.0, 0.2], v2:[0.1, 0.5, 1.0], ...}

        allevidence = {}

        # ranges = self.BNdata.binRanges
        ranges = self.BinRanges

        # loop through variables in list of inputted evidences
        for var in humanEvidence:
            # print "var is ", var
            if type(humanEvidence[var]) == list:

                input_range_min = humanEvidence[var][0]
                input_range_max = humanEvidence[var][1]

                # evidence_var = []
                allevidence[var] = [0.0] * len(ranges[var])

                # loop through bin ranges of variable "var"
                for index, binRange in enumerate(ranges[var]):

                    # print "binRange ", binRange
                    # if index == 0:
                    # this is the first bin so bin numbers larger or equal
                    # than min num and less or equal than max num (basically,
                    # include min num)
                    if input_range_min <= binRange[0] <= input_range_max or input_range_min <= binRange[1] <= input_range_max:
                        allevidence[var][index] = 1.0

                    if binRange[0] <= input_range_min <= binRange[1] or binRange[0] <= input_range_max <= binRange[1]:
                        allevidence[var][index] = 1.0

            # elif type(var) == float or type(var) == int:
            #    hard_num = float(humanEvidence[var])

        for item in allevidence:
            print item, " -- ", allevidence[item]

        return allevidence


    """
    def plotAllNodeDistributions(self, bin_ranges, binnedData, maintitle):

        # calc bin widths as per bin ranges
        trainingBinWidths = {}
        for vName in list(binnedData):
            binwidths = []
            index = 0
            for r in bin_ranges[vName]:
                binwidths.append(r[0])

                if index == len(bin_ranges[vName]) - 1:
                    binwidths.append(r[1])
                index += 1

            trainingBinWidths[vName] = binwidths

        # plot matrix
        draw_histograms(
            binnedData,
            trainingBinWidths,
            len(binnedData) / 4,
            4,
            xlabel="Ranges ",
            ylabel="Frequency",
            maintitle=maintitle,
            saveplt=True,
            displayplt=False)
    """

    """
    def plotPDs2(self,  maintitle, xlabel, ylabel, displayplt = False, **kwargs):
        '''
        plots the probability distributions
        '''

        # calculate the probability densities for the prior distributions
        binRanges = self.BNdata.binRanges
        priorPDs = {}

        bincounts = self.BNdata.bincountsDict

        for varName in bincounts:
            total = sum(sum(x) for x in bincounts[varName])
            priors = []
            for count in bincounts[varName]:
                priors.append(float(count[0]) / float(total))

            priorPDs[varName] = priors

        # plot each axes in a figure
        fig = plt.figure(figsize=(10, 10))
        fig.suptitle(maintitle, fontsize=8)
        # fig.subplots_adjust(hspace=1.0, wspace=1.0)

        i = 0
        sp = 421

        for varName in binRanges:
            # print df
            # ax = fig.add_subplot(n_rows, n_cols, i + 1)

            ax = plt.subplot(sp)

            xticksv = []
            binwidths = []
            edge = []
            for index, range in enumerate(binRanges[varName]):
                edge.append(range[0])
                binwidths.append(range[1] - range[0])
                xticksv.append(((range[1] - range[0]) / 2) + range[0])
                if index == len(binRanges[varName]) - 1: edge.append(range[1])

            # df[var_name].hist(bins = binwidths[var_name], ax=ax)
            ax.bar(xticksv, priorPDs[varName], align="center", width = binwidths, color="black", alpha=0.2)

            evidenceVars = []
            if "evidence" in kwargs:
                evidenceVars = kwargs["evidence"]

            if "posteriorPD" in kwargs:
                if len(kwargs["posteriorPD"][varName]) > 1:
                    if varName in evidenceVars:
                        ax.bar(xticksv, kwargs["posteriorPD"][varName], align="center", width = binwidths, color="green", alpha=0.2)

                    else:
                        ax.bar(xticksv, kwargs["posteriorPD"][varName], align="center", width = binwidths, color="red", alpha=0.2)

            # TODO: fix xticks .... not plotting all
            # plt.xlim(edge[0], max(edge))
            plt.xticks([round(e, 4) for e in edge], rotation="vertical")
            plt.ylim(0, 1)
            # plt.show()

            ax.grid(color="0.2", linestyle=":", linewidth=0.1, dash_capstyle="round")
            # ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            ax.set_title(varName, fontweight="bold", size=6)
            ax.set_ylabel(ylabel, fontsize=7)  # Y label
            ax.set_xlabel(xlabel, fontsize=7)  # X label
            ax.xaxis.set_tick_params(labelsize=6)
            ax.yaxis.set_tick_params(labelsize=6)

            # ax.grid(False)
            # if "xlim" in kwargs:
            #    ax.set_xlim(kwargs["xlim"][0], kwargs["xlim"][1])

            i += 1
            sp += 1

        # Improve appearance a bit.
        fig.tight_layout()

        # Add white spacing between plots and title
        fig.subplots_adjust(top=0.85)

        # if you want to set background of figure to transparent, do it here.
        # Use facecolor="none" as argument in savefig()

        if displayplt == True:
            plt.show()
    """
