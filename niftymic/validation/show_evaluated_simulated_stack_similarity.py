##
# \file show_evaluated_simulated_stack_similarity.py
# \brief      Script to show the evaluated similarity between simulated stack
#             from obtained reconstruction and original stack.
#
# This function takes the result of evaluate_simulated_stack_similarity.py as
# input.
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       November 2017
#

# Import libraries
import SimpleITK as sitk
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from natsort import natsorted

import pysitk.python_helper as ph
from nsol.similarity_measures import SimilarityMeasures as \
    SimilarityMeasures

from niftymic.utilities.input_arparser import InputArgparser
import niftymic.base.exceptions as exceptions
from niftymic.definitions import REGEX_FILENAMES


def main():

    input_parser = InputArgparser(
        description="Script to show the evaluated similarity between "
        "simulated stack from obtained reconstruction and original stack. "
        "This function takes the result of "
        "evaluate_simulated_stack_similarity.py as input. "
        "Provide --dir-output in order to save the results."
    )
    input_parser.add_dir_input(required=True)
    input_parser.add_dir_output(required=False)

    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    if not ph.directory_exists(args.dir_input):
        raise exceptions.DirectoryNotExistent(args.dir_input)

    # --------------------------------Read Data--------------------------------
    pattern = "Similarity_(" + REGEX_FILENAMES + ")[.]txt"
    p = re.compile(pattern)
    dic_filenames = {
        p.match(f).group(1): p.match(f).group(0)
        for f in os.listdir(args.dir_input) if p.match(f)
    }

    dic_stacks = {}
    for filename in dic_filenames.keys():
        path_to_file = os.path.join(args.dir_input, dic_filenames[filename])

        # Extract evaluated measures written as header in second line
        measures = open(path_to_file).readlines()[1]
        measures = re.sub("#\t", "", measures)
        measures = re.sub("\n", "", measures)
        measures = measures.split("\t")

        # Extract errors
        similarities = np.loadtxt(path_to_file, skiprows=2)

        # Build dictionary holding all similarity information for stack
        dic_stack_similarity = {
            measures[i]: similarities[:, i] for i in range(len(measures))
        }
        # dic_stack_similarity["measures"] = measures

        # Store information of to dictionary
        dic_stacks[filename] = dic_stack_similarity

    # -----------Visualize stacks individually per similarity measure----------
    ctr = [0]
    N_stacks = len(dic_stacks)
    N_measures = len(measures)
    rows = 2 if N_measures < 6 else 3
    filenames = natsorted(dic_stacks.keys(), key=lambda y: y.lower())

    for i, filename in enumerate(filenames):
        fig = plt.figure(ph.add_one(ctr))
        fig.clf()

        for m, measure in enumerate(measures):
            ax = plt.subplot(rows, np.ceil(N_measures/float(rows)), m+1)

            y = dic_stacks[filename][measure]
            x = range(1, y.size+1)
            lines = plt.plot(x, y)
            line = lines[0]
            line.set_linestyle("")
            line.set_marker(ph.MARKERS[0])
            # line.set_markerfacecolor("w")
            plt.xlabel("Slice")
            plt.ylabel(measure)
            ax.set_xticks(x)

            if measure in ["SSIM", "NCC"]:
                ax.set_ylim([0, 1])

        plt.suptitle(filename)
        try:
            # Open windows (and also save them) in full screen
            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()
        except:
            pass
        plt.show(block=False)
        if args.dir_output is not None:
            filename = "Similarity_%s.pdf" % filename
            ph.save_fig(fig, args.dir_output, filename)

    # -----------All in one (meaningful in case of similar scaling)----------
    fig = plt.figure(ph.add_one(ctr))
    fig.clf()
    data = {}
    for m, measure in enumerate(measures):
        for i, filename in enumerate(filenames):
            similarities = dic_stacks[filename][measure]
            labels = [filename] * similarities.size
            if m == 0:
                if "Stack" not in data.keys():
                    data["Stack"] = labels
                else:
                    data["Stack"] = np.concatenate((data["Stack"], labels))
            if measure not in data.keys():
                data[measure] = similarities
            else:
                data[measure] = np.concatenate(
                    (data[measure], similarities))
    df_melt = pd.DataFrame(data).melt(
        id_vars="Stack",
        var_name="",
        value_name=" ",
        value_vars=measures,
    )
    ax = plt.subplot(1, 1, 1)
    b = sns.boxplot(
        data=df_melt,
        hue="Stack",  # different colors for different "Stack"
        x="",
        y=" ",
        order=measures,
    )
    ax.set_axisbelow(True)
    try:
        # Open windows (and also save them) in full screen
        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()
    except:
        pass
    plt.show(block=False)
    if args.dir_output is not None:
        filename = "Boxplot.pdf"
        ph.save_fig(fig, args.dir_output, filename)

    # # -------------Boxplot: Plot individual similarity measures v1----------
    # for m, measure in enumerate(measures):
    #     fig = plt.figure(ph.add_one(ctr))
    #     fig.clf()
    #     data = {}
    #     for i, filename in enumerate(filenames):
    #         similarities = dic_stacks[filename][measure]
    #         labels = [filename] * similarities.size
    #         if "Stack" not in data.keys():
    #             data["Stack"] = labels
    #         else:
    #             data["Stack"] = np.concatenate((data["Stack"], labels))
    #         if measure not in data.keys():
    #             data[measure] = similarities
    #         else:
    #             data[measure] = np.concatenate(
    #                 (data[measure], similarities))
    #     df_melt = pd.DataFrame(data).melt(
    #         id_vars="Stack",
    #         var_name="",
    #         value_name=measure,
    #     )
    #     ax = plt.subplot(1, 1, 1)
    #     b = sns.boxplot(
    #         data=df_melt,
    #         hue="Stack",  # different colors for different "Stack"
    #         x="",
    #         y=measure,
    #     )
    #     ax.set_axisbelow(True)
    #     plt.show(block=False)

    # # -------------Boxplot: Plot individual similarity measures v2----------
    # for m, measure in enumerate(measures):
    #     fig = plt.figure(ph.add_one(ctr))
    #     fig.clf()
    #     data = {}
    #     for i, filename in enumerate(filenames):
    #         similarities = dic_stacks[filename][measure]
    #         labels = [filename] * len(filenames)
    #         if filename not in data.keys():
    #             data[filename] = similarities
    #         else:
    #             data[filename] = np.concatenate(
    #                 (data[filename], similarities))
    #     for filename in filenames:
    #         data[filename] = pd.Series(data[filename])
    #     df = pd.DataFrame(data)
    #     df_melt = df.melt(
    #         var_name="",
    #         value_name=measure,
    #         value_vars=filenames,
    #     )
    #     ax = plt.subplot(1, 1, 1)
    #     b = sns.boxplot(
    #         data=df_melt,
    #         x="",
    #         y=measure,
    #         order=filenames,
    #     )
    #     ax.set_axisbelow(True)
    #     plt.show(block=False)

    return 0


if __name__ == '__main__':
    main()
