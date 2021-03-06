# -*- coding: utf-8 -*-
#
# This file is part of SIDEKIT.
#
# SIDEKIT is a python package for speaker verification.
# Home page: http://www-lium.univ-lemans.fr/sidekit/
#
# SIDEKIT is a python package for speaker verification.
# Home page: http://www-lium.univ-lemans.fr/sidekit/
#
# SIDEKIT is free software: you can redistribute it and/or modify
# it under the terms of the GNU LLesser General Public License as
# published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# SIDEKIT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with SIDEKIT.  If not, see <http://www.gnu.org/licenses/>.

"""
Copyright 2014-2019 Anthony Larcher

:mod:`sidekit_io` provides methods to read and write from and to different
formats.
"""

import copy
import numpy
import torch

__license__ = "LGPL"
__author__ = "Anthony Larcher"
__copyright__ = "Copyright 2014-2019 Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'


def znorm(enrol_test_scores, enrol_imp_scores, sym=False):
    """
    Apply Z-norm to a set of scores
    imp_scores are formed by scoring enrollment utterance with all files from the impostor cohort
    thus: enrol_test_scores.modelset and enrol_imp_scores.modelset must be the same

    This function assumes that all models from enrol_test_scores are in enrol_imp_scores

    :param sym:
    :param enrol_test_scores: a score object of trials between enrolment data and test data
    :param enrol_imp_scores: a score object of trials between enrolment data and impostor data
    :return: a scores object of normalized scores
    """
    # Align enrol_test_scores.modelset and enrol_imp_scores.modelset
    scores_znorm = copy.deepcopy(enrol_test_scores)
    scores_znorm.sort()
    enrol_imp_scores.sort()

    # Compute the new enrol_test_scores normalized scores
    if sym:
        mean_per_model = (enrol_imp_scores.scoremat.sum(1) - numpy.diag(enrol_imp_scores.scoremat)) / (enrol_imp_scores.scoremat.shape[1] - 1)
        tmp = numpy.square(enrol_imp_scores.scoremat - mean_per_model)
        std_per_model = (tmp.sum(1) - numpy.diag(tmp)) / (tmp.shape[1] - 1)
    else:
        mean_per_model = enrol_imp_scores.scoremat.mean(1)
        std_per_model = enrol_imp_scores.scoremat.std(1)
    scores_znorm.scoremat = (scores_znorm.scoremat - mean_per_model) / std_per_model

    return scores_znorm


def tnorm(enrol_test_scores, imp_test_scores):
    """
    Apply t-normalization on a set of scores

    :param enrol_test_scores: a score object of trials between enrolment data and test data
    :param imp_test_scores:a score object of trials between impostor data and test data
    :return: the normalized scores
    """
    # Align enrol_test_scores.segset and imp_test_scores.segset
    scores_tnorm = copy.deepcopy(enrol_test_scores)
    scores_tnorm.sort()
    imp_test_scores.sort()

    # Compute the new enrol_test_scores normalized scores
    mean_per_segment = imp_test_scores.scoremat.mean(0)
    std_per_segment = imp_test_scores.scoremat.std(0)
    scores_tnorm.scoremat = (scores_tnorm.scoremat - mean_per_segment) / std_per_segment

    return scores_tnorm


def ztnorm(enrol_test_scores, enrol_imp_scores, imp_test_scores, imp_imp_scores):
    """
    Apply a sequence of z-norm then t-norm

    :param enrol_test_scores: a score object of trials between enrolment data and test data
    :param enrol_imp_scores: a score object of trials between enrolment data and impostor data
    :param imp_test_scores: a score object of trials between impostor data and test data
    :param imp_imp_scores: a score object of trials between impostor data and impostor data
    :return: the normalized scores
    """

    # Apply Z-norm first on enrol_test_scores by using enrol_imp_scores
    # and on imp_test_scores by using imp_imp_scores
    #
    # to produce Z_enrol_test_scores and Z_imp_test_scores
    z_enrol_test_scores = znorm(enrol_test_scores, enrol_imp_scores)
    z_imp_test_scores = znorm(imp_test_scores, imp_imp_scores, sym=True)

    # Apply t-norm on Z_enrol_test_scores with Z_imp_test_scores
    zt_enrol_test_scores = tnorm(z_enrol_test_scores, z_imp_test_scores)

    return zt_enrol_test_scores


def asnorm(enrol_xv, cohort_xv, ndx):
    """

    :param enrol_test_scores: a score object of trials between enrolment data and test data
    :return:the normalized scores
    """
    # Compute cosine similarity
    enrol_test_scores = torch.einsum('ij,kj', enrol_xv, enrol_xv).numpy()
    cohort_xv = torch.nn.functional.normalize(cohort_xv, dim=1)



    calib_scores = torch.einsum('ij,kj', enrol_xv, cohort_xv)
    topk_cohort = calib_scores.topk(200, dim=1).values
    calib_mean = topk_cohort.mean(dim=1).numpy()
    calib_std = topk_cohort.std(dim=1).numpy()

    s_enrol_test_scores = 0.5 * ((enrol_test_scores.T - calib_mean) / calib_std).T +\
                          0.5 * (enrol_test_scores - calib_mean) / calib_std

    return s_enrol_test_scores


