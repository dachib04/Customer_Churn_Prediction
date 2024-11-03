# imports.py
# This file consolidates all necessary imports for the project for easier management.

# Data manipulation and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Model selection and evaluation
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance

# Data balancing
from imblearn.over_sampling import SMOTE


# Utility for handling zip files
import zipfile
