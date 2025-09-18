"""
Contains a streamlit app allowing exploration of the contour tree's arcs and how well they cover the underlying dataset.

Usage:
streamlit run vis/app.py <ct_dir> <landscapes_dir> <datasets_dir>
"""

from sys import argv
import os

import pyct as ct
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

from experiment import Dataset, LossLandscapeExperiment, find_all_datasets, find_all_experiments

def main():
    if len(argv) != 4:
        print("Usage: streamlit run vis/app.py <ct_dir> <landscapes_dir> <datasets_dir>")
        exit(1)

    ct_dir = argv[1].strip("/\\")
    landscapes_dir = argv[2].strip("/\\")
    datasets_dir = argv[3].strip("/\\")

    st.title("Loss Landscape Arc Explorer")

    datasets = find_all_datasets(datasets_dir)
    experiments = find_all_experiments(datasets, landscapes_dir, ct_dir)

    print(f"Found {len(datasets)} datasets and {len(experiments)} experiments")
    print("\n".join([repr(exp) for exp in experiments]))

    

if __name__ == "__main__":
    main()
