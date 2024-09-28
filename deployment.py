import streamlit as st
import time
import pandas as pd
import plotly.express as px
import matplotlib.pylot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import io
from sklearn.preprocessing 

print("Run Success")

# ------- FUNCTION -------#
def data_load(datas):
    import pandas as pd
    data = pd.read_csv(datas)
    return data

#Fungsi pengecekan apakah format yang diupload csv
def is_csv(filename):
    return filename.lower().endswith
    