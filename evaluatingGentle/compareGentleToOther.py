import os
from pathlib import Path
from io import StringIO
import shutil
import subprocess
import json
from collections import defaultdict, Counter
import statistics
import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn

import parseTextGrid


########################################################################################
# Main method.
########################################################################################


def main():
    # p = "/home/tyler/Documents/ProgrammyStuff/gentle/evaluatingGentle/Sergey_Dataset"
    p = "/home/tyler/Documents/ProgrammyStuff/gentle/evaluatingGentle/TIMIT_Dataset"
    createReportComparingGentleAndOther(p)


########################################################################################
# Reporting helpers.
########################################################################################


def createReportComparingGentleAndOther(rootDir, reportsDir="./comparisonReports"):
    """Go through every folder in a directory and look at all the Gentle output JSON
    files (which should have had an associated .textGrid file generated to compare
    Gentle and Other's phone timings.) Make charts describing the aggregate results of
    all the comparisons. These charts report on stuff like average timing diff,
    distribution of timing diffs, what phones did Gentle have trouble with, which phones
    were replaced for others, etc.

    :param str rootDir: Path to root of directory structure to traverse and find
        all the Gentle output and .textGrid files.
    :param str reportsDir: Absolute or relative path (relative to where the script is
        run from) to where we want to save the reports (.png's of charts, etc.)

    No return value. Instead, save a bunch of files to disk.
    """

    #####
    ## Aggregation
    #####

    allExtraPhonesByPhone = Counter()
    allMissingPhonesByPhone = Counter()
    allStartDiffsByPhone = defaultdict(list)
    allEndDiffsByPhone = defaultdict(list)
    allReplacementCountsByPhone = defaultdict(Counter)

    numSentencesIncluded = 0
    numSentencesExcluded = 0

    for path, dirs, files in os.walk(rootDir):
        for filename in files:
            if filename.endswith("_gentlePhoneTimings.json"):
                filenameBase = filename.split("_gentlePhoneTimings.json")[0]
                textGridFilename = f"{filenameBase}.textGrid"
                textGridFilepath = os.path.join(path, textGridFilename)
                try:
                    fileResults = compareGentleAndOtherTextGridTiers(textGridFilepath)
                except TextGridException:
                    continue

                extraPhones = fileResults["extraPhones"]
                missingPhones = fileResults["missingPhones"]
                startDiffs = fileResults["startDiffs"]
                endDiffs = fileResults["endDiffs"]
                replacementCounts = fileResults["replacementCounts"]
                containsOOV = fileResults["containsOOV"]

                # # Exclude sentences if
                # # - there was an "out-of-vocabulary" word
                # # - the phone matching process had more problems than matches
                # numExtraPhones = sum(extraPhones.values())
                # numMissingPhones = sum(missingPhones.values())
                # numMatchingPhones = sum(len(l) for l in endDiffs.values())
                # tooManyProblems = numExtraPhones + numMissingPhones > numMatchingPhones
                # skipSentence = containsOOV or tooManyProblems
                # if skipSentence:
                #     numSentencesExcluded += 1
                #     continue

                # Otherwise, include the results from this sentence in our aggregates.
                numSentencesIncluded += 1
                allExtraPhonesByPhone += extraPhones
                allMissingPhonesByPhone += missingPhones
                for phone, diffs in startDiffs.items():
                    allStartDiffsByPhone[phone].extend(diffs)
                for phone, diffs in endDiffs.items():
                    allEndDiffsByPhone[phone].extend(diffs)
                for phone, counts in replacementCounts.items():
                    allReplacementCountsByPhone[phone] += counts

    #####
    ## Visuals
    #####

    try:
        os.makedirs(reportsDir)
    except FileExistsError:
        pass

    figureIdx = 0

    ## Histogram of all phones' start time diffs.
    figureIdx += 1
    startDiffsHistFig = plt.figure(figureIdx)
    startDiffsHistAxes = startDiffsHistFig.add_subplot(111)
    allStartDiffs = [
        d["diff"] for dList in allStartDiffsByPhone.values() for d in dList
    ]
    allStartDiffsAbs = [abs(d) for d in allStartDiffs]
    avgStartDiff = statistics.mean(allStartDiffsAbs)
    medianStartDiff = np.percentile(allStartDiffsAbs, 50)
    prct99StartDiff = np.percentile(allStartDiffsAbs, 99)
    phoneTimingHistBinWidth = 0.005
    startDiffsHistPlot = seaborn.histplot(
        allStartDiffsAbs,
        binwidth=phoneTimingHistBinWidth,
        ax=startDiffsHistAxes,
    )
    startDiffsHistPlot.set_title("All Phones Start Time Diffs")
    startDiffsHistPlot.set_xlabel("time (s)")
    startDiffsText = f"Average: {round(avgStartDiff, 3)}\nMedian: {round(medianStartDiff, 3)}\n99%: {round(prct99StartDiff, 3)}"
    startDiffsHistAxes.text(
        0.5,
        0.5,
        startDiffsText,
        transform=startDiffsHistAxes.transAxes,
    )
    plt.savefig(f"{reportsDir}/allPhones_startDiffs.png")
    plt.close(startDiffsHistFig)

    ## Histogram of all phones' end time diffs.
    figureIdx += 1
    endDiffsHistFig = plt.figure(figureIdx)
    endDiffsHistAxes = endDiffsHistFig.add_subplot(111)
    allEndDiffs = [d["diff"] for dList in allEndDiffsByPhone.values() for d in dList]
    allEndDiffsAbs = [abs(d) for d in allEndDiffs]
    avgEndDiff = statistics.mean(allEndDiffsAbs)
    medianEndDiff = np.percentile(allEndDiffsAbs, 50)
    prct99EndDiff = np.percentile(allEndDiffsAbs, 99)
    endDiffsHistPlot = seaborn.histplot(
        allEndDiffsAbs,
        binwidth=phoneTimingHistBinWidth,
        ax=endDiffsHistAxes,
    )
    endDiffsHistPlot.set_title("All Phones End Time Diffs")
    endDiffsHistPlot.set_xlabel("time (s)")
    endDiffsText = f"Average: {round(avgEndDiff, 3)}\nMedian: {round(medianEndDiff, 3)}\n99%: {round(prct99EndDiff, 3)}"
    endDiffsHistAxes.text(
        0.5,
        0.5,
        endDiffsText,
        transform=endDiffsHistAxes.transAxes,
    )
    plt.savefig(f"{reportsDir}/allPhones_endDiffs.png")
    plt.close(endDiffsHistFig)

    ## Histograms of each individual phone's start time diffs.
    for phone in allStartDiffsByPhone:
        figureIdx += 1
        phoneStartDiffsFig = plt.figure(figureIdx)
        phoneStartDiffsHistAxes = phoneStartDiffsFig.add_subplot(111)
        phoneStartDiffs = [d["diff"] for d in allStartDiffsByPhone[phone]]
        phoneStartDiffsAbs = [abs(d) for d in phoneStartDiffs]
        avgPhoneStartDiff = statistics.mean(phoneStartDiffsAbs)
        medianPhoneStartDiff = np.percentile(phoneStartDiffsAbs, 50)
        prct99PhoneStartDiff = np.percentile(phoneStartDiffsAbs, 99)
        phoneStartDiffsHistPlot = seaborn.histplot(
            phoneStartDiffsAbs,
            binwidth=phoneTimingHistBinWidth,
            ax=phoneStartDiffsHistAxes,
        )
        phoneStartDiffsHistPlot.set_title(f"{phone} Start Time Diffs")
        phoneStartDiffsHistPlot.set_xlabel("time (s)")
        phoneStartDiffsText = f"Average: {round(avgPhoneStartDiff, 3)}\nMedian: {round(medianPhoneStartDiff, 3)}\n99%: {round(prct99PhoneStartDiff, 3)}"
        phoneStartDiffsHistAxes.text(
            0.5,
            0.5,
            phoneStartDiffsText,
            transform=phoneStartDiffsHistAxes.transAxes,
        )
        plt.savefig(f"{reportsDir}/{phone}_startDiffs.png")
        plt.close(phoneStartDiffsFig)

    ## Histograms of each individual phone's end time diffs.
    for phone in allEndDiffsByPhone:
        figureIdx += 1
        phoneEndDiffsFig = plt.figure(figureIdx)
        phoneEndDiffsHistAxes = phoneEndDiffsFig.add_subplot(111)
        phoneEndDiffs = [d["diff"] for d in allEndDiffsByPhone[phone]]
        phoneEndDiffsAbs = [abs(d) for d in phoneEndDiffs]
        avgPhoneEndDiff = statistics.mean(phoneEndDiffsAbs)
        medianPhoneEndDiff = np.percentile(phoneEndDiffsAbs, 50)
        prct99PhoneEndDiff = np.percentile(phoneEndDiffsAbs, 99)
        phoneEndDiffsHistPlot = seaborn.histplot(
            phoneEndDiffsAbs,
            binwidth=phoneTimingHistBinWidth,
            ax=phoneEndDiffsHistAxes,
        )
        phoneEndDiffsHistPlot.set_title(f"{phone} End Time Diffs")
        phoneEndDiffsHistPlot.set_xlabel(f"time (s)")
        phoneEndDiffsText = f"Average: {round(avgPhoneEndDiff, 3)}\nMedian: {round(medianPhoneEndDiff, 3)}\n99%: {round(prct99PhoneEndDiff, 3)}"
        phoneEndDiffsHistAxes.text(
            0.5,
            0.5,
            phoneEndDiffsText,
            transform=phoneEndDiffsHistAxes.transAxes,
        )
        plt.savefig(f"{reportsDir}/{phone}_endDiffs.png")
        plt.close(phoneEndDiffsFig)

    ## Summary table of info about each phone symbol. (Does not include which phones
    ## matched which phones how often. That is a separate table.)
    figureIdx += 1
    summaryTableFig = plt.figure(figureIdx)
    summaryTableAxes = summaryTableFig.add_subplot(111)
    # Each row of the table will represent a symbol.
    allSymbols = (
        set(allStartDiffsByPhone)
        | set(allMissingPhonesByPhone)
        | set(allExtraPhonesByPhone)
    )
    rowLabels = sorted(allSymbols)
    # Each column is a piece of info about a symbol (avg start diff, etc.).
    totalColLabel = "Total"
    matchedColLabel = "Matched"
    missingColLabel = "Unmatched"
    extraColLabel = "Extra"
    medianStartDiffColLabel = "Start Diff"
    medianStartDiffAbsColLabel = "Start Diff (abs)"
    medianEndDiffColLabel = "End Diff"
    medianEndDiffAbsColLabel = "End Diff (abs)"
    colLabels = [
        totalColLabel,
        matchedColLabel,
        missingColLabel,
        extraColLabel,
        medianStartDiffColLabel,
        medianStartDiffAbsColLabel,
        medianEndDiffColLabel,
        medianEndDiffAbsColLabel,
    ]
    coloredCols = {
        medianStartDiffAbsColLabel,
        medianEndDiffAbsColLabel,
    }
    # For each row, create it by filling in each column's value for that row.
    summaryTableDataDict = {}
    for symbol in allSymbols:

        startDiffs = [d["diff"] for d in allStartDiffsByPhone[symbol]]
        endDiffs = [d["diff"] for d in allEndDiffsByPhone[symbol]]
        startDiffsAbs = [abs(d) for d in startDiffs]
        endDiffsAbs = [abs(d) for d in endDiffs]

        rowData = {}

        numMatched = len(startDiffs)
        numMissing = allMissingPhonesByPhone[symbol]
        numTotal = numMatched + numMissing
        percentMatched = round((numMatched / numTotal) * 100) if numTotal > 0 else 0
        percentMissing = round((numMissing / numTotal) * 100) if numTotal > 0 else 0
        numExtra = allExtraPhonesByPhone[symbol]
        rowData[totalColLabel] = {"value": numTotal}
        rowData[matchedColLabel] = {"value": numMatched, "percent": percentMatched}
        rowData[missingColLabel] = {"value": numMissing, "percent": percentMissing}
        rowData[extraColLabel] = {"value": numExtra}

        medianStartDiffAbs = np.percentile(startDiffsAbs, 50) if startDiffsAbs else 0
        medianStartDiff = np.percentile(startDiffs, 50) if startDiffs else 0
        medianEndDiffAbs = np.percentile(endDiffsAbs, 50) if endDiffsAbs else 0
        medianEndDiff = np.percentile(endDiffs, 50) if endDiffs else 0
        rowData[medianStartDiffAbsColLabel] = {"value": round(medianStartDiffAbs, 3)}
        rowData[medianStartDiffColLabel] = {"value": round(medianStartDiff, 3)}
        rowData[medianEndDiffAbsColLabel] = {"value": round(medianEndDiffAbs, 3)}
        rowData[medianEndDiffColLabel] = {"value": round(medianEndDiff, 3)}

        summaryTableDataDict[symbol] = rowData
    # Get the range of each column, to help us color each column's cells appropriately.
    colExtremaForColoring = {}
    for colHeader in colLabels:
        colData = [rowData[colHeader] for rowData in summaryTableDataDict.values()]
        colValues = [d["percent"] if "percent" in d else d["value"] for d in colData]
        colExtremaForColoring[colHeader] = {
            "start": min(colValues),
            "end": max(colValues),
        }
    # Given the dict of table data, convert it to inputs to matplotlib's `table`.
    cellTextArray = []  # 2D array of data in the table
    cellColoursArray = []  # 2D array of background colors of cells in the table
    for symbol in sorted(summaryTableDataDict):
        rowData = summaryTableDataDict[symbol]
        # Add the row data (each column's value for that phone).
        cellTextRow = []
        for colHeader in colLabels:
            cellData = rowData[colHeader]
            cellValue = cellData["value"]
            cellPercent = cellData.get("percent")
            cellText = f"{cellValue:.3f}" if cellValue < 1 else f"{cellValue}"
            if cellPercent is not None:
                cellText += f" ({cellPercent}%)"
            cellTextRow.append(cellText)
        cellTextArray.append(cellTextRow)
        # Add the cell colors (only certain columns are colored. colors based on the
        # cell's value relative to other values in that column).
        cellColourRow = []
        for colHeader in colLabels:
            cellColour = (1, 1, 1)
            if colHeader in coloredCols:
                cellData = rowData[colHeader]
                endForColoring = colExtremaForColoring[colHeader]["end"]
                if cellData["value"] == endForColoring:
                    cellColour = (0.8, 0.8, 0.9)
            cellColourRow.append(cellColour)
        cellColoursArray.append(cellColourRow)
    # Create the matplotlib table.
    summaryTable = summaryTableAxes.table(
        cellText=cellTextArray,
        cellColours=cellColoursArray,
        rowLabels=rowLabels,
        colLabels=colLabels,
        loc="center",
    )
    summaryTable.scale(2, 1.5)
    summaryTable.auto_set_font_size(False)
    summaryTable.set_fontsize(13)
    summaryTableAxes.set_axis_off()
    plt.savefig(f"{reportsDir}/summaryTable.png", bbox_inches="tight")
    plt.close(summaryTableFig)

    ## Table of which phones in Other matched each phone in Gentle how many times
    ## ("Replacements Table").
    figureIdx += 1
    replacementsTableFig = plt.figure(figureIdx)
    replacementsTableAxes = replacementsTableFig.add_subplot(111)
    # Each row of the table will represent a symbol.
    rowLabels = sorted(allSymbols)
    # Each column of the table will represent a symbol (how many times that symbol in
    # Other was found as a match for a phone in Gentle), except the last column: total
    # times a particular phone in Gentle was successfully matched to phones in Other.
    rowLabelOnRightCol = ""
    colLabels = sorted(allSymbols) + [rowLabelOnRightCol]
    # Create the inputs to matplotlib's `table`.
    cellTextArray = []  # 2D array of data in the table
    cellColoursArray = []  # 2D array of background colors of cells in the table
    for gSymbol in sorted(allSymbols):
        # Add the row data (each column's value for that phone), and the cell colors.
        cellTextRow = []
        cellColourRow = []

        startDiffs = [d["diff"] for d in allStartDiffsByPhone[gSymbol]]
        numMatched = len(startDiffs)

        for tSymbol in sorted(allSymbols):
            # Add the number (and percent) of times the Other phone was found as a match
            # for the Gentle phone.
            cellValue = allReplacementCountsByPhone[gSymbol][tSymbol]
            cellPercent = round((cellValue / numMatched) * 100) if numMatched > 0 else 0
            cellText = f"{cellPercent}"
            cellTextRow.append(cellText)
            # Color the cell according to the percent (darker is higher).
            cellColour = (
                getRepresentativeColor(
                    cellPercent,
                    0,
                    100,
                    startColor=(0.9, 0.9, 1),
                    endColor=(0.7, 0.7, 0.9),
                )
                if cellPercent > 0
                else (1, 1, 1)
            )
            cellColourRow.append(cellColour)
        # Add another column of the row labels on the right for convenience.
        cellTextRow.append(gSymbol)
        cellColourRow.append((1, 1, 1))  # white
        # Append the finished row to the overall table.
        cellTextArray.append(cellTextRow)
        cellColoursArray.append(cellColourRow)
    # Create the matplotlib table.
    replacementsTable = replacementsTableAxes.table(
        cellText=cellTextArray,
        cellColours=cellColoursArray,
        rowLabels=rowLabels,
        colLabels=colLabels,
        loc="center",
    )
    replacementsTable.scale(5, 1)
    replacementsTable.auto_set_font_size(False)
    replacementsTable.set_fontsize(11)
    replacementsTableAxes.set_axis_off()
    plt.savefig(f"{reportsDir}/replacementsTable.png", bbox_inches="tight")
    plt.close(replacementsTableFig)

    ## Scatter plot of start time diffs by phone length.
    figureIdx += 1
    startDiffsByPhoneLengthFig = plt.figure(figureIdx)
    startDiffsByPhoneLengthAxes = startDiffsByPhoneLengthFig.add_subplot(111)
    startDiffsByPhoneLengthX = []
    startDiffsByPhoneLengthY = []
    for dList in allStartDiffsByPhone.values():
        for d in dList:
            phoneLength = d["phoneLength"] + (random.random() - 0.5) * 0.01
            startDiff = abs(d["diff"])
            # Exclude outliers, to help us view a representative distribution
            if startDiff < 0.08:
                startDiffsByPhoneLengthX.append(phoneLength)
                startDiffsByPhoneLengthY.append(startDiff)
    # Make the scatter plot a little sparser, to help us view a representative
    # distribution.
    startDiffsByPhoneLengthData = list(
        zip(startDiffsByPhoneLengthX, startDiffsByPhoneLengthY)
    )

    startDiffsByPhoneLengthSparse = random.sample(
        startDiffsByPhoneLengthData, min(500, len(startDiffsByPhoneLengthData))
    )
    startDiffsByPhoneLengthX, startDiffsByPhoneLengthY = zip(
        *startDiffsByPhoneLengthSparse
    )
    startDiffsByPhoneLengthAxes.scatter(
        startDiffsByPhoneLengthX, startDiffsByPhoneLengthY, s=1
    )
    startDiffsByPhoneLengthAxes.set_title(
        "Start Time Diffs By Phone Length (rand, no outliers)"
    )
    startDiffsByPhoneLengthAxes.set_xlabel("phone length (s)")
    startDiffsByPhoneLengthAxes.set_ylabel("start time diff (s)")
    plt.savefig(f"{reportsDir}/startDiffsByPhoneLength.png")
    plt.close(startDiffsByPhoneLengthFig)

    ## Scatter plot of end time diffs by phone length.
    figureIdx += 1
    endDiffsByPhoneLengthFig = plt.figure(figureIdx)
    endDiffsByPhoneLengthAxes = endDiffsByPhoneLengthFig.add_subplot(111)
    endDiffsByPhoneLengthX = []
    endDiffsByPhoneLengthY = []
    for dList in allEndDiffsByPhone.values():
        for d in dList:
            phoneLength = d["phoneLength"] + (random.random() - 0.5) * 0.01
            endDiff = abs(d["diff"])
            # Exclude outliers, to help us view a representative distribution
            if endDiff < 0.08:
                endDiffsByPhoneLengthX.append(phoneLength)
                endDiffsByPhoneLengthY.append(endDiff)
    # Make the scatter plot a little sparser, to help us view a representative
    # distribution.
    endDiffsByPhoneLengthData = list(
        zip(endDiffsByPhoneLengthX, endDiffsByPhoneLengthY)
    )
    endDiffsByPhoneLengthSparse = random.sample(
        endDiffsByPhoneLengthData, min(500, len(endDiffsByPhoneLengthData))
    )
    endDiffsByPhoneLengthX, endDiffsByPhoneLengthY = zip(*endDiffsByPhoneLengthSparse)
    endDiffsByPhoneLengthAxes.scatter(
        endDiffsByPhoneLengthX, endDiffsByPhoneLengthY, s=1
    )
    endDiffsByPhoneLengthAxes.set_title(
        "End Time Diffs By Phone Length (rand, no outliers)"
    )
    endDiffsByPhoneLengthAxes.set_xlabel("phone length (s)")
    endDiffsByPhoneLengthAxes.set_ylabel("end time diff (s)")
    plt.savefig(f"{reportsDir}/endDiffsByPhoneLength.png")
    plt.close(endDiffsByPhoneLengthFig)

    ## Text file of misc info.
    with open(f"{reportsDir}/miscInfo.txt", "wt") as f:
        f.write(f"Recordings included: {numSentencesIncluded}\n")
        f.write(f"Recordings excluded: {numSentencesExcluded}\n")


class TextGridException(Exception):
    pass


def compareGentleAndOtherTextGridTiers(filepath):
    """For a TextGrid file with a tier for Gentle timings and a tier for Other timings
    (such as TIMIT), find nearby matches between the phones in each tier and get the
    differences in start and end timings.

    :param str filepath: Absolute or relative path to a .textGrid file. The TextGrid
        file should have a tier for Gentle timings and for Other timings.

    :return dict results: Dict with the following keys:
        - extraPhones: Counter, mapping a string key (the phone) to an int (how many
            times it was extra)
        - missingPhones: Counter, mapping a string key (the phone) to an int (how many
            times it was missing)
        - startDiffs: dict, mapping a string key (the phone) to a list of dicts with
            keys
            - "diff" (start time diff between Gentle and Other, in seconds)
            - "phoneLength" (phone length, in seconds)
            ## TODO: could add more fields here to slice and dice afterward, maybe even
            ##  throw into a SQL db for flexible querying (maybe something besides a SQL
            ##  db if there's something even more convenient for quickly making charts
            ##  of various slices)
            - "isFirstPhone"
            - "isLastPhone"
            - "isVowel"
            Each dict represents a single instance of a phone in Gentle.
        - endDiffs: dict, same as `startDiffs` but for the end timings of phones
        - replacementCounts: dict, mapping a string key (the phone in Gentle) to a
            Counter mapping a string key (the phone in Other) to how many times the
            phone in Other was matched to the phone in Gentle
        - containsOOV: bool, whether or not there was a word "out-of-vocabulary" for
            Gentle and thus the matching of phones between Gentle and Other is more
            likely to get messed up.
    """
    try:
        textGrid = parseTextGrid.TextGrid.load(filepath)
    except IndexError:
        print(
            f"Couldn't parse TextGrid file: {filepath}. Probably no phones in a tier."
        )
        raise TextGridException

    # Some phones in Gentle will be either replaced or split into multiple in Other.
    # This dict maps some phones to potential replacements or additions that should
    # still be considered as "matching". For example, closure intervals of stops are
    # distinguised from the stop release in Other, so if looking for "t" and we find
    # "tcl" and then "t", we effectively combine them.
    VALID_SYMBOL_REPLACEMENTS = defaultdict(
        set,
        {
            # closure intervals of stops
            "b": {"bcl"},
            "d": {"dcl", "dx"},
            "g": {"gcl"},
            "p": {"pcl"},
            "t": {"tcl", "dx", "q"},
            "k": {"kcl"},
            "jh": {"dcl", "y"},
            "ch": {"tcl"},
            "th": {"dh", "t"},
            "zh": {"jh", "sh"},
            "z": {"s"},
            "m": {"em"},
            "n": {"nx", "en"},
            "ng": {"n", "nx"},
            "l": {"el"},
            "r": {"axr", "er"},
            "hh": {"hv"},
            "hv": {"hh"},
            "y": {"jh"},
            "aa": {"ao", "ah"},
            "ah": {"aa", "ao", "ax", "ax-h", "uh", "ih", "ix", "el", "en", "em"},
            "ao": {"aa", "ah"},
            "ih": {"ax", "ax-h", "ix", "iy", "uh", "el"},
            "iy": {"ix"},
            "eh": {"ae", "ih", "el", "axr", "er"},
            "ae": {"eh"},
            "uh": {"ax", "ix", "ux", "axr", "er"},
            "ow": {"uh"},
            "oy": {"ao", "ow"},
            "uw": {"ux"},
            "er": {"axr", "r"},
        },
    )

    # Symbols which are vowels (and thus when we encounter mix-ups, we keep track but
    # still continue with the sentence).
    VOWEL_PHONES = {
        "iy",
        "ih",
        "eh",
        "ey",
        "ae",
        "aa",
        "aw",
        "ay",
        "ah",
        "ao",
        "oy",
        "ow",
        "uh",
        "uw",
        "ux",
        "er",
        "ax",
        "ix",
        "axr",
        "ax-h",
    }

    # Symbols in the International Phonetic Alphabet mapped to their corresponding
    # 2-symbol representations in ARPABET.
    IPA_TO_ARPABET_MAP = {
        "ɑ": "aa",
        "æ": "ae",
        "ʌ": "ah",
        "ɔ": "ao",
        "ɒ": "ao",
        "aʊ": "aw",
        "ə": "ax",
        "ɚ": "axr",
        "aɪ": "ay",
        "ɛ": "eh",
        "e": "eh",
        "ɝ": "er",
        "eɪ": "ey",
        "ɪ": "ih",
        "ɨ": "ix",
        "i": "iy",
        "oʊ": "ow",
        "ɔɪ": "oy",
        "ʊ": "uh",
        "u": "uw",
        "ʉ": "ux",
        "b": "b",
        "tʃ": "ch",
        "d": "d",
        "ð": "dh",
        "ɾ": "dx",
        "l̩": "el",
        "m̩": "em",
        "n̩": "en",
        "f": "f",
        "ɡ": "g",
        "h": "hh",
        "dʒ": "jh",
        "k": "k",
        "l": "l",
        "m": "m",
        "n": "n",
        "ŋ": "ng",
        "ɾ̃": "nx",
        "p": "p",
        "ʔ": "q",
        "ɹ": "r",
        "s": "s",
        "ʃ": "sh",
        "t": "t",
        "θ": "th",
        "v": "v",
        "w": "w",
        "ʍ": "wh",
        "j": "y",
        "z": "z",
        "ʒ": "zh",
    }

    # Symbols which, if found while not looking for them, may be ignored without being
    # considered "extra".
    MAY_IGNORE_FROM_OTHER = {
        # TIMIT
        "pau",  # pause
        "epi",  # epenthetic silence
        "h#",  # begin/end marker (non-speech events)
        # Sergey
        "",
    }

    # For a given Gentle phone, as we scan through Other phones trying to match to it,
    # don't consider any Other phones this many seconds away from the Gentle phone.
    MAX_DIFF_sec = 0.1

    # Assign the tiers to either Gentle or Other.
    otherTier = None
    gentleTier = None
    for tier in textGrid:
        if "gentle" in tier.nameid.lower():
            gentleTier = tier
        else:
            otherTier = tier
    if otherTier is None or gentleTier is None:
        raise Exception("TextGrid must have tiers for Gentle and Other timings.")

    # Initialize vars to keep track of results.
    extraPhones = Counter()
    missingPhones = Counter()
    startDiffs = defaultdict(list)
    endDiffs = defaultdict(list)
    replacementCounts = defaultdict(Counter)
    containsOOV = False

    ## ALGORITHM
    ## - Keep track of the latest phone in the Other phones which has been matched to a
    ##      Gentle phone.
    ## - Start iterating through Gentle phones.
    ## - For each Gentle phone, start with the latest matched Other phone (so sometimes
    ##      an Other phone could match to multiple Gentle phones) and iterate through
    ##      the Other phones until they start (and subsequently stop) matching the
    ##      current Gentle phone. Keep track of the start/end timings before moving on
    ##      to the next Gentle phone.
    ## - During this, also keep track of Gentle phones not successfully matched
    ##      ("missing") and Other phones not successfully matched ("extra").

    # The index of the last Other phone which has been matched to a Gentle phone.
    lastOtherIdxMatched = 0

    numGentlePhones = len(gentleTier.simple_transcript)

    # Iterate through Gentle phones.
    for gIdx, (gPhoneStart, gPhoneEnd, gPhoneSymbol) in enumerate(
        gentleTier.simple_transcript
    ):
        gPhoneStart, gPhoneEnd = float(gPhoneStart), float(gPhoneEnd)
        gPhoneLength = gPhoneEnd - gPhoneStart
        # Gentle symbols are like "t_B", "t_I", or "t_E", where all we need is the "t".
        gPhoneSymbol = gPhoneSymbol.split("_")[0]

        # Note if we find an "out-of-vocabulary" symbol in Gentle's labeling.
        if gPhoneSymbol == "oov":
            containsOOV = True

        # Starting with the last Other phone that matched the previous Gentle phone
        # (possible for an Other phone to match multiple Gentle phones), iterate through
        # Other phones until we start (and subsequently stop) matching the current
        # Gentle phone.
        oPhoneIdx = lastOtherIdxMatched
        hasStartedMatching = False
        lastMatchingPhoneEnd = None
        while oPhoneIdx < len(otherTier.simple_transcript):
            oPhoneStart, oPhoneEnd, oPhoneSymbol = otherTier.simple_transcript[
                oPhoneIdx
            ]
            oPhoneStart, oPhoneEnd = float(oPhoneStart), float(oPhoneEnd)
            # If the labeling used an IPA symbol, map it to ARPABET for consistency with
            # Gentle.
            if oPhoneSymbol in IPA_TO_ARPABET_MAP:
                oPhoneSymbol = IPA_TO_ARPABET_MAP[oPhoneSymbol]

            # If the current Other phone's timing is too far before the Gentle phone's
            # timing,
            if oPhoneEnd < gPhoneStart - MAX_DIFF_sec:
                # Move on to the next Other phone immediately.
                oPhoneIdx += 1
                continue

            # If we get too far past the Gentle phone's timing without finding a
            # matching Other phone yet,
            if oPhoneStart > gPhoneEnd + MAX_DIFF_sec and not hasStartedMatching:
                # Move on to the next Gentle phone.
                break

            # Check if the current Other phone matches the current Gentle phone (or any
            # symbols that we consider valid replacements for this Gentle phone).
            matchingSymbols = {gPhoneSymbol}
            matchingSymbols = matchingSymbols | VALID_SYMBOL_REPLACEMENTS[gPhoneSymbol]
            isMatch = oPhoneSymbol in matchingSymbols

            # If Other's phone matches,
            if isMatch:
                # Save the Other phone end time in case this Other phone ends up being
                # the end of Other's matching of the current Gentle phone.
                lastMatchingPhoneEnd = oPhoneEnd
                # If this is the first Other phone to match the current Gentle phone,
                if not hasStartedMatching:
                    # Mark down that Other started matching the current Gentle phone.
                    hasStartedMatching = True
                    # Keep track of the start time diff between Gentle and Other.
                    startDiff = gPhoneStart - oPhoneStart
                    startDiffs[gPhoneSymbol].append(
                        {"diff": startDiff, "phoneLength": gPhoneLength}
                    )
                    # Add the Other phones between the previous match and this match to
                    # the counts of extra phones.
                    for idx in range(lastOtherIdxMatched + 1, oPhoneIdx):
                        symbol = otherTier.simple_transcript[idx][2]
                        if symbol not in MAY_IGNORE_FROM_OTHER:
                            extraPhones[symbol] += 1
                # Update our count of how many times the current Other phone's symbol
                # has been matched to the current Gentle phone's symbol.
                replacementCounts[gPhoneSymbol][oPhoneSymbol] += 1
                # Update our index of the last Other phone that has been matched to a
                # Gentle phone.
                lastOtherIdxMatched = oPhoneIdx

            # If Other's phone does not match but we had already started matching the
            # current Gentle phone (indicating this is the end of Other's matching of
            # the current Gentle phone),
            if not isMatch and hasStartedMatching:
                # Keep track of the end time diff.
                endDiff = gPhoneEnd - lastMatchingPhoneEnd
                endDiffs[gPhoneSymbol].append(
                    {"diff": endDiff, "phoneLength": gPhoneLength}
                )
                # Move on to the next Gentle phone.
                break

            # Move to the next Other phone.
            oPhoneIdx += 1

        # Once we've ended (or broke out of) the loop, if Other has not started matching
        # the current Gentle phone, mark down that this Gentle phone was missing.
        if not hasStartedMatching:
            missingPhones[gPhoneSymbol] += 1

    # Once we've gotten through the Gentle phones, make sure any unmatched trailing
    # Other phones are marked down as being extra.
    for idx in range(lastOtherIdxMatched + 1, len(otherTier.simple_transcript)):
        symbol = otherTier.simple_transcript[idx][2]
        if symbol not in MAY_IGNORE_FROM_OTHER:
            extraPhones[symbol] += 1

    return {
        "extraPhones": extraPhones,
        "missingPhones": missingPhones,
        "startDiffs": startDiffs,
        "endDiffs": endDiffs,
        "replacementCounts": replacementCounts,
        "containsOOV": containsOOV,
    }


def getRepresentativeColor(
    value, startValue, endValue, startColor=(1, 1, 1), endColor=(0.7, 0.7, 0.9)
):
    """Given a numeric value, get a color that represents that value relative to a range
    of numbers which map to a range of colors.

    :param float value: Numeric value to represent with the returned color.
    :param float startValue: Numeric value represented by `startColor` (probably either
        the min or max of the dataset `value` comes from).
    :param float endValue: Numeric value represented by `endColor` (probably either the
        min or max of the dataset `value` comes from).
    :param (float, float, float) startColor: RGB 3-tuple of floats between 0 and 1 for
        the color representing `startValue`.
    :param (float, float, float) endColor: RGB 3-tuple of floats between 0 and 1 for the
        color representing `endValue`.

    :return (float, float, float) color: RGB 3-tuple of floats between 0 and 1 for the
        color we give the passed in `value`.
    """
    proportionFromStart = (value - startValue) / (endValue - startValue)
    proportionFromStart = max(0, proportionFromStart)
    proportionFromStart = min(1, proportionFromStart)
    color = tuple(
        startRGBVal + (proportionFromStart * (endRGBVal - startRGBVal))
        for startRGBVal, endRGBVal in zip(startColor, endColor)
    )
    return color


########################################################################################
# File manipulation helpers.
########################################################################################


def convertTimitPhnAndGentleJsonToTextGrid(inputDir, filename):
    """Take TIMIT and Gentle phone timings, and create a .textGrid file so they may be
    compared to each other in Praat.

    TIMIT
    =====

    TIMIT dataset outputs phone timings in a format like:

    0 2260 h#
    2260 4070 sh
    4070 5265 iy
    ...

    with each number being a number of samples in the associated .wav file, and the
    sample rate being 16kHz (so divide each number by 16,000 to get seconds).

    Gentle
    =====

    The Gentle tool outputs a json file whose format is best observed by running the
    Gentle tool on an audio file and looking at the resulting json file. (The format is
    from a gentle.Transcription object's .to_json() function.)

    -----

    :path str inputDir: Absolute or relative path to a directory containing TIMIT data
        and Gentle result data, with matching names (see the `filename` parameter).
    :path str filename: Part of the filename which is common across the audio file,
        TIMIT phone timings file, and Gentle phone timings file, e.g. "SA1" (which is in
        SA1.wav, SA1.PHN, and SA1_gentlePhoneTimings.json)

    No return value. Instead, a new file, named the same as the input file but with a
    .textGrid file extension, is created on disk. .textGrid format described here:
    https://www.fon.hum.uva.nl/praat/manual/TextGrid_file_formats.html.
    """

    # Phone timings from TIMIT dataset that we are converting to a .textGrid file.
    timitPhnFilepath = os.path.join(inputDir, f"{filename}.PHN")
    with open(timitPhnFilepath, "r") as f:
        timitPhoneTimings = f.readlines()

    # Phone timings from Gentle that we are converting to a .textGrid file.
    gentleJsonFilepath = os.path.join(inputDir, f"{filename}_gentlePhoneTimings.json")
    with open(gentleJsonFilepath, "r") as f:
        gentleResult = json.load(f)

    # To build up contents of new .textGrid file
    outputStrIO = StringIO()

    # File header stuff.
    outputStrIO.write('File type = "ooTextFile"\n')
    outputStrIO.write('Object class = "TextGrid"\n')
    outputStrIO.write("\n")

    # Once-per-file data.

    TIMIT_WAV_SAMPLE_RATE = 16000
    timitTotalTimeLength = int(timitPhoneTimings[-1].split()[1]) / TIMIT_WAV_SAMPLE_RATE
    gentleTotalTimeLength = max(word.get("end", 0) for word in gentleResult["words"])
    totalTimeLength = max(timitTotalTimeLength, gentleTotalTimeLength)

    outputStrIO.write("xmin = 0\n")
    outputStrIO.write(f"xmax = {totalTimeLength}\n")
    outputStrIO.write("tiers? <exists>\n")
    outputStrIO.write("size = 2\n")
    outputStrIO.write("item []:\n")

    # TIMIT tier: header data.

    timitTotalPhones = len(timitPhoneTimings)

    outputStrIO.write("    item [1]:\n")
    outputStrIO.write('        class = "IntervalTier"\n')
    outputStrIO.write('        name = "TIMIT phones"\n')
    outputStrIO.write("        xmin = 0\n")
    outputStrIO.write(f"        xmax = {timitTotalTimeLength}\n")
    outputStrIO.write(f"        intervals: size = {timitTotalPhones}\n")

    # TIMIT tier: Once-per-phone data.

    for phoneIdx, phoneLine in enumerate(timitPhoneTimings):
        phoneStart_samples, phoneEnd_samples, phoneSymbol = phoneLine.split()
        phoneStart = int(phoneStart_samples) / TIMIT_WAV_SAMPLE_RATE
        phoneEnd = int(phoneEnd_samples) / TIMIT_WAV_SAMPLE_RATE

        outputStrIO.write(f"        intervals [{phoneIdx + 1}]:\n")
        outputStrIO.write(f"            xmin = {phoneStart}\n")
        outputStrIO.write(f"            xmax = {phoneEnd}\n")
        outputStrIO.write(f'            text = "{phoneSymbol}"\n')

    # Gentle tier: header data.

    gentleTotalPhones = sum(
        len(word.get("phones", [])) for word in gentleResult["words"]
    )

    outputStrIO.write("    item [2]:\n")
    outputStrIO.write('        class = "IntervalTier"\n')
    outputStrIO.write('        name = "Gentle phones"\n')
    outputStrIO.write("        xmin = 0\n")
    outputStrIO.write(f"        xmax = {gentleTotalTimeLength}\n")
    outputStrIO.write(f"        intervals: size = {gentleTotalPhones}\n")

    # Gentle tier: Once-per-phone data.

    phoneIdx = 1
    for wordObj in gentleResult["words"]:
        if wordObj["case"] != "success":
            continue
        phoneStart = wordObj["start"]
        for phoneObj in wordObj["phones"]:
            phoneSymbol = phoneObj["phone"]
            phoneDuraction = phoneObj["duration"]
            phoneEnd = phoneStart + phoneDuraction

            outputStrIO.write(f"        intervals [{phoneIdx}]:\n")
            outputStrIO.write(f"            xmin = {phoneStart}\n")
            outputStrIO.write(f"            xmax = {phoneEnd}\n")
            outputStrIO.write(f'            text = "{phoneSymbol}"\n')

            # Update loop variables for the next phone.
            phoneStart = phoneEnd
            phoneIdx += 1

    # Write to the output .textGrid file.
    outputFilepath = os.path.join(inputDir, f"{filename}.textGrid")
    with open(outputFilepath, "w+") as f:
        outputStrIO.seek(0)
        shutil.copyfileobj(outputStrIO, f)


def convertTimitPhnToTextGrid(filepath):
    """TIMIT dataset outputs phone timings in a format like

    0 2260 h#
    2260 4070 sh
    4070 5265 iy
    ...

    with each number being a number of samples in the associated .wav file, and the
    sample rate being 16kHz (so divide each number by 16,000 to get seconds).

    This conversion function reads the TIMIT .PHN file specified and outputs a
    .textGrid file, whose format is described here:
    https://www.fon.hum.uva.nl/praat/manual/TextGrid_file_formats.html.

    :path str filepath: Absolute or relative path to a .PHN file in the TIMIT dataset.

    No return value. Instead, a new file, named the same as the input file but with a
    .textGrid file extension, is created on disk.
    """
    TIMIT_WAV_SAMPLE_RATE = 16000

    # Phone timings from TIMIT dataset that we are converting to a .textGrid file.
    with open(filepath, "r") as f:
        phoneTimings = f.readlines()

    # To build up contents of a new .textGrid file.
    outputStrIO = StringIO()

    # File header stuff.
    outputStrIO.write('File type = "ooTextFile"\n')
    outputStrIO.write('Object class = "TextGrid"\n')
    outputStrIO.write("\n")

    # Once-per-file data.

    totalTimeLength = int(phoneTimings[-1].split()[1]) / TIMIT_WAV_SAMPLE_RATE
    totalPhones = len(phoneTimings)

    outputStrIO.write("xmin = 0\n")
    outputStrIO.write(f"xmax = {totalTimeLength}\n")
    outputStrIO.write("tiers? <exists>\n")
    outputStrIO.write("size = 1\n")
    outputStrIO.write("item []:\n")
    outputStrIO.write("    item [1]:\n")
    outputStrIO.write('        class = "IntervalTier"\n')
    outputStrIO.write('        name = "phones"\n')
    outputStrIO.write("        xmin = 0\n")
    outputStrIO.write(f"        xmax = {totalTimeLength}\n")
    outputStrIO.write(f"        intervals: size = {totalPhones}\n")

    # Once-per-phone data.

    for phoneIdx, phoneLine in enumerate(phoneTimings):
        phoneStart_samples, phoneEnd_samples, phoneSymbol = phoneLine.split()
        phoneStart = int(phoneStart_samples) / TIMIT_WAV_SAMPLE_RATE
        phoneEnd = int(phoneEnd_samples) / TIMIT_WAV_SAMPLE_RATE

        outputStrIO.write(f"        intervals [{phoneIdx}]:\n")
        outputStrIO.write(f"            xmin = {phoneStart}\n")
        outputStrIO.write(f"            xmax = {phoneEnd}\n")
        outputStrIO.write(f'            text = "{phoneSymbol}"\n')

    # Write the contents to the output .textGrid file.
    outputFilepath = str(Path(filepath).with_suffix(".textGrid"))
    with open(outputFilepath, "w+") as f:
        outputStrIO.seek(0)
        shutil.copyfileobj(outputStrIO, f)


def convertGentleJsonToTextGrid(filepath):
    """The Gentle tool outputs a json file whose format is best observed by running the
    Gentle tool on an audio file and looking at the resulting json file. (The format is
    from a gentle.Transcription object's .to_json() function.)
    This conversion function reads the json file specified and outputs a .textGrid file,
    whose format is described here:
    https://www.fon.hum.uva.nl/praat/manual/TextGrid_file_formats.html.

    :path str filepath: Absolute or relative path to a file output by Gentle to json.

    No return value. Instead, a new file, named the same as the input file but with a
    .textGrid file extension, is created on disk.
    """
    # Phone timings from Gentle that we are converting to a .textGrid file.
    with open(filepath, "r") as f:
        gentleResult = json.load(f)

    # To build up contents of new .textGrid file
    outputStrIO = StringIO()

    # File header stuff.
    outputStrIO.write('File type = "ooTextFile"\n')
    outputStrIO.write('Object class = "TextGrid"\n')
    outputStrIO.write("\n")

    # Once-per-file data.

    totalTimeLength = max(word.get("end", 0) for word in gentleResult["words"])
    totalPhones = sum(len(word.get("phones", [])) for word in gentleResult["words"])

    outputStrIO.write("xmin = 0\n")
    outputStrIO.write(f"xmax = {totalTimeLength}\n")
    outputStrIO.write("tiers? <exists>\n")
    outputStrIO.write("size = 1\n")
    outputStrIO.write("item []:\n")
    outputStrIO.write("    item [1]:\n")
    outputStrIO.write('        class = "IntervalTier"\n')
    outputStrIO.write('        name = "phones"\n')
    outputStrIO.write("        xmin = 0\n")
    outputStrIO.write(f"        xmax = {totalTimeLength}\n")
    outputStrIO.write(f"        intervals: size = {totalPhones}\n")

    # Once-per-phone data.

    phoneIdx = 1
    for wordObj in gentleResult["words"]:
        if wordObj["case"] != "success":
            continue
        phoneStart = wordObj["start"]
        for phoneObj in wordObj["phones"]:
            phoneSymbol = phoneObj["phone"]
            phoneDuraction = phoneObj["duration"]
            phoneEnd = phoneStart + phoneDuraction

            outputStrIO.write(f"        intervals [{phoneIdx}]:\n")
            outputStrIO.write(f"            xmin = {phoneStart}\n")
            outputStrIO.write(f"            xmax = {phoneEnd}\n")
            outputStrIO.write(f'            text = "{phoneSymbol}"\n')

            # Update loop variables for the next phone.
            phoneStart = phoneEnd
            phoneIdx += 1

    # Write the contents to the output .textGrid file.
    outputFilepath = str(Path(filepath).with_suffix(".textGrid"))
    with open(outputFilepath, "w+") as f:
        outputStrIO.seek(0)
        shutil.copyfileobj(outputStrIO, f)


def renameTimitFilesToLowercaseExtensions(timitDirpath):
    """Because the phone segmenation tool assumes lowercase .wav file extension, but
    TIMIT has uppercase .WAV file extensions, rename their audio files to use lowercase.

    :param str timitDirpath: Path to root of directory structure to traverse and find
        all the .WAV files to rename.

    No return value. Instead, every .WAV file found will be renamed on disk to use .wav
        instead.
    """
    for path, dirs, files in os.walk(timitDirpath):
        for filename in files:
            if filename.endswith(".WAV") or filename.endswith(".TXT"):
                oldFilepath = os.path.join(path, filename)
                oldExtension = filename[-4:]
                newExtension = oldExtension.lower()
                newFilename = f"{filename[:-4]}{newExtension}"
                newFilepath = os.path.join(path, newFilename)
                os.rename(oldFilepath, newFilepath)


def removeNumbersFromTimitTranscripts(timitDirpath):
    """TIMIT .txt transcripts start with something like '0 54362 ' but we don't want
    those numbers in the transcripts we input to Gentle since they are not spoken in the
    audio. Go through TIMIT and remove these starting numbers from all the transcripts.

    :param str timitDirpath: Path to root of directory structure to traverse and find
        all the .txt files to modify.

    No return value. Instead, replace the content of every .txt file with the same
        sentence but without the starting couple numbers and spaces.
    """
    for path, dirs, files in os.walk(timitDirpath):
        for filename in files:
            if filename.endswith(".txt") or filename.endswith(".TXT"):
                filenameBase = filename[:-4]
                phnFilename = f"{filenameBase}.PHN"
                isTranscript = phnFilename in files
                if isTranscript:
                    filepath = os.path.join(path, filename)
                    with open(filepath, "r+") as f:
                        content = f.read()
                        contentWords = content.split(" ")
                        # If the first two words are just numbers, remove them from the
                        # transcript.
                        if contentWords[0].isdigit() and contentWords[1].isdigit():
                            newContent = " ".join(contentWords[2:])
                            f.seek(0)
                            f.write(newContent)
                            f.truncate()


def runPhoneSegmentationOnTimit(timitDirpath):
    """Go through every folder within TIMIT and run Gentle to get phone timings. See
    phoneSegmentation.py for details, but basically for each directory it will look for
    all audio files and for any that have .txt transcripts of the appropriate filename
    it will run Gentle and create a .json file of phone timings.

    :param str timitDirpath: Path to root of directory structure to traverse and find
        all the .WAV files to rename.

    No return value. Instead, create a bunch of .json files on disk in the same folders
        that the audio and transcript files are found in.
    """
    for path, dirs, files in os.walk(timitDirpath):
        subprocess.run(["python3", "phoneSegmentation.py", path])


def createTextGridsFromTimitPhnAndGentleJsonForAllTimit(timitDirpath):
    """Go through every folder within TIMIT and create .textGrid files which have a tier
    for both Gentle and TIMIT, so they may be compared in Praat (or programatically).

    :param str timitDirpath: Path to root of directory structure to traverse and find
        all the phone timings files to make .textGrid files from.

    No return value. Instead, create a bunch of .textGrid files on disk in the same
        folders that the audio and transcript files are found in.
    """
    for path, dirs, files in os.walk(timitDirpath):
        for filename in files:
            if filename.endswith("_gentlePhoneTimings.json"):
                filenameBase = filename.split("_gentlePhoneTimings.json")[0]
                convertTimitPhnAndGentleJsonToTextGrid(path, filenameBase)


def createDirOfWavAndTxtAndSergeyTextgrids(sergeyDataDirpath):
    """Take a folder of Sergey's collected + hand-segmented T5/T11 data, and organize
    files to be passed to our Gentle script.

    :param str sergeyDataDirpath: Path to root of directory structure to find the T5/T11
        data (both the phone hand-segmented phone timings in .textGrid files, and the
        recorded .wav files).

    No return value. Instead, for each dir in the specified root dir, create a new dir
        inside it that contains all the recorded .wav files, new .txt transcript files,
        and hand-segmented phone timing .textGrid files. One of each of those types of
        files for each recording .wav file, named correspondingly.
    """
    for sessionDirname in os.listdir(sergeyDataDirpath):
        sessionDirpath = os.path.join(sergeyDataDirpath, sessionDirname)

        if not os.path.isdir(sessionDirpath):
            continue

        wavFilesDirpath = os.path.join(sessionDirpath, "wav_files")
        textGridFilesDirpath = os.path.join(sessionDirpath, "textGrid_files")
        # Add a new folder next to the existing ones, where we'll organize all the files
        # so it's easy for our Gentle script.
        combinedFilesDirpath = os.path.join(sessionDirpath, "combined_files")
        try:
            os.mkdir(combinedFilesDirpath)
        except OSError:
            pass

        for blockDirname in os.listdir(wavFilesDirpath):
            blockDirpath = os.path.join(wavFilesDirpath, blockDirname)

            if not os.path.isdir(blockDirpath):
                continue

            for wavFilename in os.listdir(blockDirpath):
                # Ignore non-audio files.
                if not wavFilename.endswith(".wav"):
                    continue

                # `filename` has no path, no extension.
                filename = Path(wavFilename).stem
                # The Sergey hand-segmented .TextGrid file uses "_" where there are "."
                # in the .wav filename. Let's use that as the base filename in the new
                # `combined_files/` directory.
                filename = filename.replace(".", "_")

                wavFilepath = os.path.join(blockDirpath, wavFilename)
                newWavFilename = f"{filename}.wav"
                newWavFilepath = os.path.join(combinedFilesDirpath, newWavFilename)
                shutil.copy(wavFilepath, newWavFilepath)

                # The Sergey hand-segmented .TextGrid file uses "_" where there are "."
                # in the .wav filename.
                textGridFilename = f"{filename}.TextGrid"
                textGridFilepath = os.path.join(textGridFilesDirpath, textGridFilename)
                if not os.path.exists(textGridFilepath):
                    print(
                        f"Could not find corresponding .TextGrid file for "
                        f"{wavFilepath} at {textGridFilepath}"
                    )
                newTextGridFilename = f"{filename}_sergeyPhoneTimings.textGrid"
                newTextGridFilepath = os.path.join(
                    combinedFilesDirpath, newTextGridFilename
                )
                shutil.copy(textGridFilepath, newTextGridFilepath)

                newTxtFilename = f"{filename}.txt"
                newTxtFilepath = os.path.join(combinedFilesDirpath, newTxtFilename)
                # The word spoken in the recording is at the end of the filename.
                word = filename.split("_")[-1]
                # Write just the spoken word to the transcript .txt file.
                with open(newTxtFilepath, "w") as f:
                    f.write(f"{word}\n")


def convertSergeyTextgridAndGentleJsonToTextGrid(inputDir, filename):
    """Take Gentle and Sergey's hand-done phone timings, and create a .textGrid file so
    they may be compared to each other in Praat.

    Sergey
    =====

    Sergey's hand-done phone timings are already in TextGrid format, the same as this
    function outputs.

    Gentle
    =====

    The Gentle tool outputs a json file whose format is best observed by running the
    Gentle tool on an audio file and looking at the resulting json file. (The format is
    from a gentle.Transcription object's .to_json() function.)

    -----

    :path str inputDir: Absolute or relative path to a directory containing Sergey's
        hand-done data and Gentle result data, with matching names (see the `filename`
        parameter).
    :path str filename: Part of the filename which is common across the audio file,
        Sergey's hand-done phone timings file, and Gentle phone timings file, e.g.
        "R_t5_2019_01_23_B1_trial0001_fawn" (which is in
        R_t5_2019_01_23_B1_trial0001_fawn.wav,
        R_t5_2019_01_23_B1_trial0001_fawn_sergeyPhoneTimings.textGrid, and
        R_t5_2019_01_23_B1_trial0001_fawn_gentlePhoneTimings.json)

    No return value. Instead, a new file, named the same as the input file but with a
    .textGrid file extension, is created on disk. .textGrid format described here:
    https://www.fon.hum.uva.nl/praat/manual/TextGrid_file_formats.html.
    """

    # Phone timings from Sergey's hand-done dataset.
    sergeyTextgridFilepath = os.path.join(
        inputDir, f"{filename}_sergeyPhoneTimings.textGrid"
    )
    try:
        with open(sergeyTextgridFilepath, "r", encoding="utf-8") as f:
            sergeyTextgridLines = f.readlines()
    except UnicodeDecodeError:
        with open(sergeyTextgridFilepath, "r", encoding="utf-16") as f:
            sergeyTextgridLines = f.readlines()

    # Phone timings from Gentle that we are converting to a .textGrid file.
    gentleJsonFilepath = os.path.join(inputDir, f"{filename}_gentlePhoneTimings.json")
    with open(gentleJsonFilepath, "r") as f:
        gentleResult = json.load(f)

    # To build up contents of new .textGrid file
    outputStrIO = StringIO()

    # For the once-per-file stuff and the first tier (Sergey's hand-done tier), mostly
    # copy Sergey's .textGrid file, except replace a few parts:
    # - xmax should be the max of Sergey's and Gentle's xmax values.
    # - size should be 2 (2 tiers, Sergey's and Gentle's) instead of 1
    # - name of Sergey's tier should be more than just "phones" since there's 2 tiers
    for line in sergeyTextgridLines:
        if line.startswith("xmax ="):
            sergeyTotalTimeLength = float(line.split()[2])
            gentleTotalTimeLength = max(
                word.get("end", 0) for word in gentleResult["words"]
            )
            totalTimeLength = max(sergeyTotalTimeLength, gentleTotalTimeLength)
            outputStrIO.write(f"xmax = {totalTimeLength}\n")
        elif line.startswith("size ="):
            outputStrIO.write("size = 2\n")
        elif line.startswith("        name ="):
            outputStrIO.write('        name = "Sergey hand-done phones"\n')
        else:
            outputStrIO.write(line)

    # Now do custom stuff for the second tier: Gentle's results.

    # Gentle tier: header data.

    gentleTotalPhones = sum(
        len(word.get("phones", [])) for word in gentleResult["words"]
    )

    outputStrIO.write("    item [2]:\n")
    outputStrIO.write('        class = "IntervalTier"\n')
    outputStrIO.write('        name = "Gentle phones"\n')
    outputStrIO.write("        xmin = 0\n")
    outputStrIO.write(f"        xmax = {gentleTotalTimeLength}\n")
    outputStrIO.write(f"        intervals: size = {gentleTotalPhones}\n")

    # Gentle tier: Once-per-phone data.

    phoneIdx = 1
    for wordObj in gentleResult["words"]:
        if wordObj["case"] != "success":
            continue
        phoneStart = wordObj["start"]
        for phoneObj in wordObj["phones"]:
            phoneSymbol = phoneObj["phone"]
            phoneDuraction = phoneObj["duration"]
            phoneEnd = phoneStart + phoneDuraction

            outputStrIO.write(f"        intervals [{phoneIdx}]:\n")
            outputStrIO.write(f"            xmin = {phoneStart}\n")
            outputStrIO.write(f"            xmax = {phoneEnd}\n")
            outputStrIO.write(f'            text = "{phoneSymbol}"\n')

            # Update loop variables for the next phone.
            phoneStart = phoneEnd
            phoneIdx += 1

    # Write to the output .textGrid file.
    outputFilepath = os.path.join(inputDir, f"{filename}.textGrid")
    with open(outputFilepath, "w+") as f:
        outputStrIO.seek(0)
        shutil.copyfileobj(outputStrIO, f)


def createTextGridsFromSergeyTextGridAndGentleJsonForWholeDir(inputDir):
    """Go through the folder of Sergey+Gentle data and create .textGrid files which have
    a tier for both Gentle and Sergey's hand-done segmentation, so they may be compared
    in Praat (or programatically).

    :param str inputDir: Path to the directory with all the phone timings files to make
        .textGrid files from.

    No return value. Instead, create a bunch of .textGrid files on disk in the same
        folders that the audio and transcript files are found in.
    """
    for path, dirs, files in os.walk(inputDir):
        for filename in files:
            if filename.endswith("_gentlePhoneTimings.json"):
                filenameBase = filename.split("_gentlePhoneTimings.json")[0]
                convertSergeyTextgridAndGentleJsonToTextGrid(path, filenameBase)


if __name__ == "__main__":
    main()
