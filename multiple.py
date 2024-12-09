#!/usr/bin/env python3
"""
Copyright 2021 Harvey Ormston

This file is part of osc_gen.

    osc_gen is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    osc_gen is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with osc_gen.  If not, see <https://www.gnu.org/licenses/>.
"""

# This example combines multiple single-cycle wav files into a WaveTable.

import argparse
import os
import sys

import numpy as np
import soundfile as sf
from wavetable_gen import example_files 

from osc_gen import visualize, wavetable, wavfile, zosc

STORE_FILES = True
SHOW_PLOTS = True

DESCRIPTION = (
    "Convert single-cycle wav files in a directory "
    + "into a wavetable. Assumes that the direcory contains "
    + "wav files at the root level. Does not look in subdirectories."
)

HELP_CYCLE_DIR = "Directory to look for wav files."
HELP_NUM_SLOTS = "Number of slots in output wavetable (default=16)."
HELP_WAVE_LEN = "Length of the output waves (default=128)."

HELP_SELECT = (
    "How to select files (first, last, even, default=first). "
    + "first: chose the first X files from a list of Y files, "
    + "last: chose the last X files from a list of Y files, "
    + "even: chose X files evenly spaced from a list of Y files."
)

HELP_SORT = (
    "How to sort files (alpha, random, reverse, default=alpha): "
    + "alpha: alphabetical sort, "
    + "random: random sort, "
    + "reverse: reverse alphabetical sort."
)

HELP_NAME = "Name of the output file."


def render(zwt, name):
    """Write to file or plot a wavetable"""

    if STORE_FILES:
        osc_path = make_osc_path()
        fname = os.path.join(osc_path, name + ".h2p")
        print(f"Saving to {fname}")
        zosc.write_wavetable(zwt, fname)
        fname = os.path.join(osc_path, name + ".wav")
        print(f"Saving to {fname}")
        wavfile.write_wavetable(zwt, fname)
    if SHOW_PLOTS:
        visualize.plot_wavetable(zwt, title=name)


def check_args(args, parser):
    """Validate arguments"""

    if args.num_slots < 1:
        parser.error("Minimum num_slots is 1")

    if args.wave_len < 2:
        parser.error("Minimum wave_len is 2")

    if not os.path.exists(args.cycle_dir):
        parser.error(f"error: Directory {args.cycle_dir} does not exist.")

    if not os.path.isdir(args.cycle_dir):
        parser.error(f"error: {args.cycle_dir} is not a directory.")

    if args.sort not in ("alpha", "reverse", "random"):
        parser.error(f"{args.sort} is not a valid sort option.")

    if args.select not in ("first", "last", "even"):
        parser.error(f"{args.select} is not a valid select option.")


def main():
    """main"""

    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("cycle_dir", help=HELP_CYCLE_DIR)
    parser.add_argument("--num_slots", default=16, type=int, help=HELP_NUM_SLOTS)
    parser.add_argument("--wave_len", default=128, type=int, help=HELP_WAVE_LEN)
    parser.add_argument("--select", default="first", help=HELP_SELECT)
    parser.add_argument("--sort", default="alpha", help=HELP_SORT)
    parser.add_argument("--name", default=None, help=HELP_NAME)
    args = parser.parse_args()

    check_args(args, parser)

    zwt = wavetable.WaveTable(args.num_slots, wave_len=args.wave_len)

    wavfiles = []

    print(f"Looking in directory {args.cycle_dir}")

    for x in os.listdir(args.cycle_dir):
        if x.lower().endswith(".wav"):
            wavfiles.append(os.path.join(args.cycle_dir, x))

    print(f"Found {len(wavfiles)} wav files.")

    if len(wavfiles) < zwt.num_slots:
        print(
            f"error: {len(wavfiles)} .wav files found in {args.cycle_dir}. "
            f"Expected at least {zwt.num_slots}"
        )
        sys.exit()

    print(f"Sorting wav files using {args.sort}.")

    if args.sort == "alpha":
        wavfiles.sort()
    elif args.sort == "reverse":
        wavfiles.sort(reverse=True)
    elif args.sort == "random":
        np.random.shuffle(wavfiles)

    print(f"Selecting wav files using {args.select}.")

    if args.select == "first":
        wavfiles = wavfiles[: zwt.num_slots]
    elif args.select == "last":
        wavfiles = wavfiles[-zwt.num_slots :]
    elif args.select == "even":
        idx = np.round(np.linspace(0, len(wavfiles) - 1, args.num_slots)).astype(int)
        wavfiles = np.array(wavfiles)[idx].tolist()

    print("Selected these files:\n")

    for fname in wavfiles:
        print("\t", fname)
    print()

    zwt.waves = [sf.read(x)[0] for x in wavfiles]

    if args.name is None:
        name = os.path.basename(args.cycle_dir)
    else:
        name = args.name

    render(zwt, name)

    print("Done!")


if __name__ == "__main__":
    main()