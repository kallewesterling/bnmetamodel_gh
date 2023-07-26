from __future__ import print_function

from .BayesianNetwork import *
from .BNdata import *
from .Helper_functions import loadDataFromCSV


class BN_Metamodel_easy:
    """
    This class is meant to build a Bayesian Network (BN) metamodel from
    provided CSV data.

    It handles the following steps in the process:
    1. Loads the CSV data
    2. Builds the BN skeleton (topology)
    3. Specifies bin types and bin numbers
    4. Prepares the data using BNdata
    5. Builds the BN with the data and skeleton

    The generated BN can then be used for inferring probabilities based on soft
    and hard evidence, as well as generate the BN itself.
    """

    def __init__(self, csvdata, targets, **kwargs):
        """
        Initialise the BN_Metamodel_easy class.

        Parameters
        ----------
        csvdata : str
            The path to the CSV data file to be used in building the BN.

        targets : list
            The list of target variables in the BN.

        **kwargs : dict, optional
            A dictionary of additional keyword arguments. Current accepted keys
            are:

            - "numBinsDict": dict, optional
                A dictionary mapping variables to the number of bins to be
                used in the BN for each variable. If not provided, the default
                setting is 7 bins for each variable.
        """
        self.targets = targets

        # load data
        self.variables = loadDataFromCSV(csvdata, True)[0]

        # declare as empty attributes
        self.binTypeDict = {}
        self.numBinsDict = {}

        # extract skeleton from csv
        BNskel = BNskelFromCSV(csvdata, targets)

        # if numBinsDict is empty:
        # if bool(BN_Metamodel_easy.numBinsDict) == False { }

        # binTypeDict = {}
        # numBinsDict = {}

        if "numBinsDict" in kwargs:
            self.numBinsDict = kwargs["numBinsDict"]

        # if "binTypeDict" in kwargs:
        for var in self.variables:
            if var in targets:
                # default: all distributions are discretized by equal spacing
                self.binTypeDict[var] = "e"

                # default: all distributions have 6 bins by default
                self.numBinsDict[var] = 7
            else:
                # default: all distributions are discretized by equal spacing
                self.binTypeDict[var] = "e"

                # default: all distributions have 6 bins by default
                self.numBinsDict[var] = 7

        # data = BNdata(csvdata, self.binTypeDict, self.numBinsDict)
        data = BNdata(
            csvdata=csvdata,
            targetlist=self.targets,
            binTypeDict=self.binTypeDict,
            numBinsDict=self.numBinsDict,
        )
        # data = BNdata(csvdata, self.targets)

        # self.binTypeDict = data.binTypeDict
        # self.numBinsDict = data.numBinsDict

        self.learnedBaynet = BayesianNetwork(BNdata=data, netStructure=BNskel)

    def json(self):
        """
        Retrieves the learned Bayesian Network in JSON format.

        This method allows the learned Bayesian Network to be exported in a
        JSON data format, which can be useful for serialization, storage, or
        transmission.

        Returns
        -------
        json_data : dict
            The learned Bayesian Network represented in JSON format.
        """
        return self.learnedBaynet.json_data

    def generate(self):
        """
        Retrieves the learned Bayesian Network object.

        This method allows the learned Bayesian Network object to be accessed
        directly, which can be useful for further manipulations or analyses
        that require the original object.

        Returns
        -------
        learnedBaynet : BayesianNetwork
            The learned Bayesian Network object.
        """
        return self.learnedBaynet

    def changeNumBinsDict(dict):
        """
        Changes the number of bins dictionary (``numBinsDict``) for the class.

        This method allows changing the number of bins for each variable in
        the Bayesian Network globally. This can be useful when you want to
        adjust the granularity of the probability distributions represented by
        the network.

        Parameters
        ----------
        new_dict : dict
            The new dictionary mapping variables to the number of bins to be
            used in the Bayesian Network for each variable.

        Returns
        -------
        None
        """
        BN_Metamodel_easy.numBinsDict = dict

    def inferPD_JT_soft(self, query, softevidence):
        """
        Performs inference on the learned Bayesian Network with soft evidence.

        This method computes posterior probabilities for the query variables,
        given the provided soft evidence. The evidence is treated as "soft",
        meaning that it is treated as observations that have uncertainty. The
        method also plots the resulting posterior distributions.

        Parameters
        ----------
        query : variable, or list of variables
            The variable(s) for which to compute the posterior distribution.

        softevidence : dict
            A dictionary mapping variables to observed values, treated as soft
            evidence.

        Returns
        -------
        posteriors : dict
            The computed posterior distributions for the query variables.
        """
        posteriors = self.learnedBaynet.inferPD_JT_soft(softevidence)

        self.learnedBaynet.plotPDs(
            xlabel="Ranges ",
            ylabel="Probability",
            maintitle="Posterior Distributions",
            displayplt=True,
            posteriorPD=posteriors,
            evidence=softevidence.keys(),
        )

        return posteriors

    def inferPD_JT_hard(self, query, hardevidence):
        """
        Performs inference on the learned Bayesian Network with hard evidence.

        This method computes posterior probabilities for the query variables,
        given the provided hard evidence. The evidence is treated as "hard",
        meaning that it is treated as exact observations without uncertainty.
        The method also plots the resulting posterior distributions.

        Parameters
        ----------
        query : variable, or list of variables
            The variable(s) for which to compute the posterior distribution.

        hardevidence : dict
            A dictionary mapping variables to observed values, treated as hard
            evidence.

        Returns
        -------
        posteriors : dict
            The computed posterior distributions for the query variables.
        """
        posteriors = self.learnedBaynet.inferPD_JT_hard(hardevidence)

        self.learnedBaynet.plotPDs(
            xlabel="Ranges ",
            ylabel="Probability",
            maintitle="Posterior Distributions",
            displayplt=True,
            posteriorPD=posteriors,
            evidence=hardevidence.keys(),
        )

        return posteriors

    def inferPD_VE_hard(self, query, evidence):
        """
        Performs inference on the learned Bayesian Network with hard evidence
        using Variable Elimination (VE).

        This method computes posterior probabilities for the query variables,
        given the provided hard evidence, by applying the Variable Elimination
        algorithm. The evidence is treated as "hard", meaning that it is
        treated as exact observations without uncertainty. The method also
        plots the resulting posterior distributions.

        Parameters
        ----------
        query : variable, or list of variables
            The variable(s) for which to compute the posterior distribution.

        evidence : dict
            A dictionary mapping variables to observed values, treated as hard
            evidence.

        Returns
        -------
        posteriors : dict
            The computed posterior distributions for the query variables.
        """
        a, posteriors = self.learnedBaynet.inferPD(query, evidence)

        self.learnedBaynet.plotPDs(
            xlabel="Ranges ",
            ylabel="Probability",
            maintitle="Posterior Distributions",
            displayplt=True,
            posteriorPD=posteriors,
            evidence=evidence.keys(),
        )

        return posteriors
