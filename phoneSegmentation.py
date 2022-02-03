"""
Usage:

python phoneSegmentation.py directoryWhereAudioFilesAre/
python phoneSegmentation.py directoryWhereAudioFilesAre/ -c path/to/config.yaml
python phoneSegmentation.py directoryWhereAudioFilesAre/ -f audiofilename.wav

The resulting phone timings will be in the same directory as the inputs, 1 json file per
audio file.
"""


import os
from pathlib import Path
import argparse
import multiprocessing

import yaml
import gentle


########################################################################################
# Main method.
########################################################################################


def main():

    ## Command-line args.

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputDir", type=str, help="directory where the audio files and transcripts are"
    )
    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        type=str,
        help="path to configuration yaml file",
    )
    parser.add_argument(
        "-f",
        "--file",
        dest="file",
        type=str,
        help="path to single audio file to segment instead of the whole directory",
    )
    args = parser.parse_args()

    ## Load config file (if specified).

    config = {}
    if args.config:
        with open(args.config, "r") as fh:
            config = yaml.load(fh)

    ## Collect list of audio files from the input directory.

    audioFileExtensions = [".wav", ".mp3"]
    if args.file:
        audioFilepath = os.path.join(args.inputDir, args.file)
        if not os.path.exists(audioFilepath):
            print(f"Did not find {audioFilepath}.")
            return
        audioFilepaths = [audioFilepath]
    else:
        audioFilepaths = [
            entry.path
            for entry in os.scandir(args.inputDir)
            if any(entry.name.endswith(ext) for ext in audioFileExtensions)
        ]
        print(f"Found {len(audioFilepaths)} audio files in {args.inputDir}.")

    ## Phonescribe the audio files.

    audioFilesPhonescribed = []
    audioFilesErrored = []
    audioFilesMissingTranscripts = []

    for audioFilepath in audioFilepaths:
        audioFilename = os.path.basename(audioFilepath)
        print(f"Phonescribing {audioFilename} ...")

        ## Check if this audio file has a corresponding transcript file.

        txtFilepath = str(Path(audioFilepath).with_suffix(".txt"))
        txtFilename = os.path.basename(txtFilepath)
        if not os.path.exists(txtFilepath):
            audioFilesMissingTranscripts.append(audioFilename)
            print(f"Missing transcript file for {audioFilename}.")
            continue

        ## Use `gentle` to get results.

        try:
            resultJson = phonescribeOneFile(audioFilepath, txtFilepath)
        except Exception as e:
            print(f"Errored: {e}")
            audioFilesErrored.append(audioFilename)
        else:
            audioFilesPhonescribed.append(audioFilename)

        ## Write results to an output json file.

        outputFileName = txtFilename.replace(".txt", "") + "_gentlePhoneTimings.json"
        outputFilepath = os.path.join(args.inputDir, outputFileName)
        with open(outputFilepath, "w+") as fh:
            fh.write(resultJson)

        print(f"Finished {audioFilename}. See result at {outputFilepath}.")

    ## Display summary.

    if len(audioFilesPhonescribed) > 0:
        print(f"\nSuccessfully phonescribed {len(audioFilesPhonescribed)} files:")
        for filename in audioFilesPhonescribed:
            print(f"\t{filename}")
    if len(audioFilesErrored) > 0:
        print(f"\nErrored trying to phonescribe {len(audioFilesErrored)} files:")
        for filename in audioFilesErrored:
            print(f"\t{filename}")
    if len(audioFilesMissingTranscripts) > 0:
        print(f"\nMissing transcripts for {len(audioFilesMissingTranscripts)} files:")
        for filename in audioFilesMissingTranscripts:
            print(f"\t{filename}")


########################################################################################
# Helpers.
########################################################################################


def phonescribeOneFile(audioFilepath, txtFilepath):
    """For a single filename (which is associated with both an audio recording and the
        text transcription), get the phone timings and return a json string representing
        the results.

    :param str audioFilename: Absolute or relative path to the audio recording.
    :param str txtFilename: Absolute or relative path to the text transcription.

    :return str resultJson: Resulting phone timings in a json string. Important fields:
        - "words": list of dicts with the below fields:
            - "words.$.word": which word, e.g. "hi".
            - "words.$.start": seconds into the audio file this word starts.
            - "words.$.end": seconds into the audio file this word ends.
            - "words.$.phones": list of dicts with the below fields:
                - "words.$.phones.$.phone": which phone, e.g. "hh_B" ("hh" is the phone,
                    "B" is for beginning of word).
                - "words.$.phones.$.duration": number of seconds this phone lasts.

        See [point to some documentation] for more detailed description of the phone
            notation.
    """
    # Read the transcript from the file.
    with open(txtFilepath, encoding="utf-8") as fh:
        transcript = fh.read()
    # Prepare to use `gentle` to transcribe.
    resources = gentle.Resources()
    aligner = gentle.ForcedAligner(
        resources,
        transcript,
        nthreads=multiprocessing.cpu_count(),
    )
    # Convert the audio recording to a .wav file.
    with gentle.resampled(audioFilepath) as wavfile:
        # Use `gentle` to transcribe.
        result = aligner.transcribe(wavfile)
    # Output the phone timing results as a json string.
    resultJson = result.to_json()

    return resultJson


if __name__ == "__main__":
    main()
