import os
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

from rdkit import Chem
from rdkit.Chem import MACCSkeys, Descriptors, rdMolDescriptors, Crippen
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


def mol_from_smiles(smi):
    if not isinstance(smi, str):
        return None
    return Chem.MolFromSmiles(smi)


def canonical(smi):
    if not isinstance(smi, str):
        return None
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol else None


_RAW = {
    "O":                  ("water",               (1.3328,0.82,0.35,0.0720,78.36,0.000,0.000), (1.09,1.17,0.18,0.00), (1.062,0.025,0.681,1.000), 63.1),
    "CO":                 ("methanol",            (1.3284,0.43,0.47,0.0226,32.66,0.000,0.000), (0.60,0.98,0.66,0.00), (0.605,0.545,0.608,0.904), 55.4),
    "CCO":                ("ethanol",             (1.3611,0.37,0.48,0.0221,24.85,0.000,0.000), (0.54,0.86,0.75,0.00), (0.400,0.658,0.633,0.783), 51.9),
    "CC#N":               ("acetonitrile",        (1.3441,0.07,0.32,0.0288,35.69,0.000,0.000), (0.75,0.19,0.40,0.00), (0.044,0.286,0.645,0.974), 45.6),
    "CS(C)=O":            ("dmso",                (1.4783,0.00,0.88,0.0428,46.83,0.000,0.000), (1.00,0.00,0.76,0.00), (0.072,0.647,0.830,1.000), 45.1),
    "ClC(Cl)Cl":          ("chloroform",          (1.4459,0.15,0.02,0.0267, 4.71,0.000,0.250), (0.58,0.44,0.00,0.50), (0.047,0.071,0.783,0.614), 39.1),
    "ClCCl":              ("dcm",                 (1.4242,0.10,0.05,0.0272, 8.93,0.000,0.286), (0.82,0.30,0.00,0.50), (0.040,0.178,0.762,0.769), 40.7),
    "C1CCCO1":            ("thf",                 (1.4072,0.00,0.55,0.0267, 7.58,0.000,0.000), (0.58,0.00,0.55,0.00), (0.000,0.591,0.714,0.634), 37.4),
    "C1CCOC1":            ("thf",                 (1.4072,0.00,0.55,0.0267, 7.58,0.000,0.000), (0.58,0.00,0.55,0.00), (0.000,0.591,0.714,0.634), 37.4),
    "CC(C)=O":            ("acetone",             (1.3587,0.04,0.49,0.0233,20.49,0.000,0.000), (0.71,0.08,0.48,0.00), (0.000,0.475,0.651,0.907), 42.2),
    "CCOC(C)=O":          ("ethyl acetate",       (1.3724,0.00,0.45,0.0232, 5.99,0.000,0.000), (0.55,0.00,0.45,0.00), (0.000,0.542,0.656,0.603), 38.1),
    "CCCCCC":             ("hexane",              (1.3749,0.00,0.00,0.0180, 1.88,0.000,0.000), (-0.11,0.00,0.00,0.00),(0.000,0.056,0.616,0.000), 31.0),
    "c1ccccc1":           ("benzene",             (1.5011,0.00,0.14,0.0289, 2.27,0.500,0.000), (0.59,0.00,0.10,1.00), (0.000,0.124,0.793,0.270), 34.3),
    "Cc1ccccc1":          ("toluene",             (1.4969,0.00,0.14,0.0285, 2.37,0.429,0.000), (0.54,0.00,0.11,1.00), (0.000,0.128,0.782,0.284), 33.9),
    "CCOCC":              ("diethyl ether",       (1.3524,0.00,0.41,0.0169, 4.24,0.000,0.000), (0.27,0.00,0.47,0.00), (0.000,0.562,0.617,0.385), 34.5),
    "C1COCCO1":           ("dioxane",             (1.4224,0.00,0.37,0.0324, 2.21,0.000,0.000), (0.55,0.00,0.37,0.00), (0.000,0.444,0.737,0.312), 36.0),
    "CN(C)C=O":           ("dmf",                 (1.4305,0.00,0.69,0.0372,37.22,0.000,0.000), (0.88,0.00,0.69,0.00), (0.031,0.613,0.759,0.977), 43.2),
    "CC(C)O":             ("isopropanol",         (1.3776,0.33,0.56,0.0209,19.26,0.000,0.000), (0.48,0.76,0.84,0.00), (0.283,0.830,0.633,0.808), 48.4),
    "CCCO":               ("1-propanol",          (1.3850,0.37,0.48,0.0237,20.52,0.000,0.000), (0.52,0.84,0.90,0.00), (0.367,0.782,0.658,0.748), 50.7),
    "CCCCO":              ("1-butanol",           (1.3993,0.37,0.48,0.0244,17.33,0.000,0.000), (0.47,0.84,0.84,0.00), (0.341,0.809,0.674,0.655), 50.2),
    "ClC(Cl)(Cl)Cl":      ("ccl4",                (1.4601,0.00,0.00,0.0266, 2.23,0.000,0.500), (0.28,0.00,0.00,0.50), (0.000,0.044,0.768,0.000), 32.4),
    "C1CCCCC1":           ("cyclohexane",         (1.4266,0.00,0.00,0.0247, 2.02,0.000,0.000), (0.00,0.00,0.00,0.00), (0.000,0.073,0.683,0.000), 30.9),
    "c1ccncc1":           ("pyridine",            (1.5095,0.00,0.52,0.0380,12.98,0.500,0.000), (0.87,0.00,0.64,1.00), (0.033,0.581,0.842,0.761), 40.2),
    "CC(O)=O":            ("acetic acid",         (1.3716,0.61,0.44,0.0267, 6.19,0.000,0.000), (0.64,1.12,0.45,0.00), (0.689,0.390,0.651,0.728), 51.7),
    "OCCO":               ("ethylene glycol",     (1.4318,0.90,0.52,0.0480,40.23,0.000,0.000), (0.92,0.90,0.52,0.00), (0.717,0.534,0.777,0.909), 56.3),
    "OCC(O)CO":           ("glycerol",            (1.4730,1.21,0.51,0.0634,42.47,0.000,0.000), (0.93,1.21,0.51,0.00), (0.653,0.340,0.828,0.812), 57.0),
    "CCC(C)=O":           ("butanone",            (1.3788,0.06,0.48,0.0237,18.11,0.000,0.000), (0.67,0.06,0.48,0.00), (0.000,0.520,0.688,0.872), 41.3),
    "CN(C)C(C)=O":        ("dma",                 (1.4380,0.00,0.78,0.0324,37.78,0.000,0.000), (0.88,0.00,0.76,0.00), (0.028,0.650,0.763,0.987), 43.7),
    "CN1CCCC1=O":         ("nmp",                 (1.4700,0.00,0.77,0.0405,32.20,0.000,0.000), (0.92,0.00,0.77,0.00), (0.024,0.613,0.812,0.954), 42.2),
    "Clc1ccccc1":         ("chlorobenzene",       (1.5241,0.00,0.07,0.0330, 5.69,0.286,0.143), (0.71,0.00,0.07,1.00), (0.000,0.182,0.833,0.537), 36.8),
    "C[N+](=O)[O-]":      ("nitromethane",        (1.3819,0.22,0.06,0.0366,35.87,0.000,0.000), (0.85,0.22,0.06,0.00), (0.066,0.236,0.710,0.954), 46.3),
    "NC=O":               ("formamide",           (1.4472,0.71,0.48,0.0568,109.5,0.000,0.000), (0.97,0.71,0.48,0.00), (0.549,0.414,0.737,1.006), 55.8),
    "CCN(CC)CC":          ("triethylamine",       (1.4003,0.00,0.71,0.0199, 2.42,0.000,0.000), (0.14,0.00,0.71,0.00), (0.000,0.930,0.615,0.139), 33.3),
    "CCCCCCC":            ("heptane",             (1.3876,0.00,0.00,0.0197, 1.92,0.000,0.000), (-0.08,0.00,0.00,0.00),(0.000,0.040,0.631,0.000), 30.9),
    "CCCCCCCC":           ("octane",              (1.3974,0.00,0.00,0.0212, 1.94,0.000,0.000), (-0.05,0.00,0.00,0.00),(0.000,0.013,0.650,0.000), 30.9),
    "CCCCCCCCCC":         ("decane",              (1.4102,0.00,0.00,0.0233, 1.99,0.000,0.000), (-0.04,0.00,0.00,0.00),(0.000,0.013,0.674,0.000), 30.9),
    "Cc1ccc(C)cc1":       ("p-xylene",            (1.4958,0.00,0.12,0.0279, 2.27,0.429,0.000), (0.43,0.00,0.12,1.00), (0.000,0.139,0.789,0.218), 33.1),
    "Cc1ccccc1C":         ("o-xylene",            (1.5055,0.00,0.12,0.0296, 2.57,0.429,0.000), (0.47,0.00,0.12,1.00), (0.000,0.165,0.793,0.282), 33.1),
    "Cc1cccc(C)c1":       ("m-xylene",            (1.4972,0.00,0.12,0.0282, 2.37,0.429,0.000), (0.47,0.00,0.12,1.00), (0.000,0.165,0.793,0.282), 33.1),
    "CC1COC(=O)O1":       ("prop carbonate",      (1.4230,0.00,0.40,0.0418,64.92,0.000,0.000), (0.83,0.00,0.40,0.00), (0.000,0.381,0.765,0.914), 46.0),
    "CN(C)P(=O)(N(C)C)N(C)C": ("hmpa",           (1.4579,0.00,1.05,0.0346,29.61,0.000,0.000), (0.87,0.00,1.05,0.00), (0.000,0.916,0.799,0.871), 40.9),
    "CC(O)CO":            ("propylene glycol",    (1.4310,0.78,0.52,0.0361,27.50,0.000,0.000), (0.73,0.78,0.52,0.00), (0.491,0.724,0.689,0.840), 51.9),
    "CCCCCO":             ("1-pentanol",          (1.4101,0.37,0.48,0.0253,15.13,0.000,0.000), (0.40,0.84,0.86,0.00), (0.302,0.860,0.687,0.587), 49.1),
    "CCCCCCO":            ("1-hexanol",           (1.4178,0.37,0.48,0.0253,13.03,0.000,0.000), (0.40,0.80,0.84,0.00), (0.315,0.879,0.680,0.577), 48.8),
    "CCCCCCCO":           ("1-heptanol",          (1.4249,0.37,0.48,0.0266,11.75,0.000,0.000), (0.40,0.79,0.83,0.00), (0.306,0.864,0.694,0.532), 48.5),
    "CCCCCCCCO":          ("1-octanol",           (1.4295,0.37,0.48,0.0271,10.34,0.000,0.000), (0.40,0.77,0.81,0.00), (0.299,0.840,0.703,0.490), 48.1),
    "CCCCCCCCCCO":        ("1-decanol",           (1.4372,0.37,0.48,0.0280, 8.10,0.000,0.000), (0.40,0.77,0.81,0.00), (0.291,0.819,0.721,0.423), 47.7),
    "OCC(F)(F)F":         ("tfe",                 (1.2907,1.51,0.00,0.0213,26.67,0.000,0.000), (0.73,1.51,0.00,0.00), (1.072,0.000,0.543,0.922), 59.8),
    "OC(C(F)(F)F)C(F)(F)F": ("hfip",             (1.2752,1.96,0.00,0.0150,16.70,0.000,0.333), (0.65,1.96,0.00,0.00), (1.361,0.000,0.479,0.949), 65.3),
    "CCCCOCCCC":          ("dibutyl ether",       (1.3992,0.00,0.45,0.0224, 3.08,0.000,0.000), (0.24,0.00,0.46,0.00), (0.000,0.637,0.657,0.288), 33.0),
    "CCCC#N":             ("butyronitrile",       (1.3842,0.00,0.36,0.0268,24.83,0.000,0.000), (0.71,0.00,0.36,0.00), (0.000,0.299,0.659,0.951), 42.5),
    "CCC#N":              ("propionitrile",       (1.3655,0.00,0.37,0.0269,27.20,0.000,0.000), (0.71,0.00,0.37,0.00), (0.000,0.299,0.634,0.966), 43.7),
    "COC(C)=O":           ("methyl acetate",      (1.3614,0.00,0.42,0.0244, 6.68,0.000,0.000), (0.60,0.00,0.42,0.00), (0.000,0.527,0.645,0.637), 38.9),
    "CCCCOC(C)=O":        ("butyl acetate",       (1.3941,0.00,0.45,0.0246, 5.07,0.000,0.000), (0.46,0.00,0.45,0.00), (0.000,0.525,0.675,0.519), 38.5),
    "CC1CCCCC1":          ("methylcyclohexane",   (1.4231,0.00,0.00,0.0235, 2.02,0.000,0.000), (-0.06,0.00,0.00,0.00),(0.000,0.060,0.683,0.000), 30.9),
    "CC(=O)C(C)(C)C":     ("pinacolone",          (1.3951,0.00,0.57,0.0233,12.01,0.000,0.000), (0.65,0.00,0.55,0.00), (0.000,0.584,0.673,0.825), 39.6),
    "O=[N+]([O-])c1ccccc1": ("nitrobenzene",      (1.5562,0.00,0.30,0.0434,34.82,0.333,0.000), (1.01,0.00,0.30,1.00), (0.000,0.240,0.900,0.873), 41.2),
    "CC(C)CCCC(C)C":      ("isooctane",           (1.3915,0.00,0.00,0.0185, 1.94,0.000,0.000), (-0.09,0.00,0.00,0.00),(0.000,0.013,0.632,0.000), 30.9),
    "[2H]O[2H]":          ("d2o",                 (1.3284,0.82,0.35,0.0720,78.36,0.000,0.000), (1.09,1.17,0.18,0.00), (1.062,0.025,0.681,1.000), 63.1),
    "[2H]OC":             ("cd3od",               (1.3284,0.43,0.47,0.0226,32.66,0.000,0.000), (0.60,0.98,0.66,0.00), (0.605,0.545,0.608,0.904), 55.4),
    "[2H]OCC":            ("etod",                (1.3611,0.37,0.48,0.0221,24.85,0.000,0.000), (0.54,0.86,0.75,0.00), (0.400,0.658,0.633,0.783), 51.9),
    "CC1CCCO1":          ("2-methylthf",         (1.4100,0.00,0.57,0.0263, 6.97,0.000,0.000), (0.53,0.00,0.57,0.00), (0.000,0.609,0.701,0.593), 37.0),
    "CC(C)(C)O":         ("tert-butanol",        (1.3878,0.42,0.93,0.0204,12.47,0.000,0.000), (0.41,0.68,1.01,0.00), (0.145,1.072,0.632,0.658), 43.9),
    "OCCOCCO":           ("diethylene glycol",   (1.4475,0.77,0.65,0.0480,31.82,0.000,0.000), (0.82,0.77,0.65,0.00), (0.580,0.639,0.777,0.882), 53.8),
    "CC(C)CO":           ("isobutanol",          (1.3955,0.37,0.48,0.0224,17.93,0.000,0.000), (0.40,0.69,0.84,0.00), (0.311,0.832,0.674,0.655), 48.6),
    "ClCCCl":            ("1,2-dichloroethane",  (1.4448,0.10,0.11,0.0317,10.37,0.000,0.333), (0.81,0.10,0.11,0.00), (0.030,0.126,0.771,0.742), 41.3),
    "CCCCC":             ("pentane",             (1.3575,0.00,0.00,0.0157, 1.84,0.000,0.000), (-0.08,0.00,0.00,0.00),(0.000,0.000,0.593,0.000), 30.9),
    "CC(C)CC(C)(C)C":    ("2,4-dimethylpentane", (1.3815,0.00,0.00,0.0181, 1.92,0.000,0.000), (-0.08,0.00,0.00,0.00),(0.000,0.013,0.628,0.000), 30.9),
    "CNC=O":             ("nmf",                 (1.4319,0.40,0.55,0.0385,182.4,0.000,0.000), (0.90,0.62,0.80,0.00), (0.374,0.549,0.759,0.972), 47.8),
    "N#Cc1ccccc1":       ("benzonitrile",        (1.5289,0.00,0.33,0.0391,25.20,0.500,0.000), (0.90,0.00,0.33,1.00), (0.000,0.281,0.851,0.895), 42.0),
    "CC(C)OC(C)C":       ("diisopropyl ether",   (1.3680,0.00,0.49,0.0185, 3.80,0.000,0.000), (0.19,0.00,0.53,0.00), (0.000,0.657,0.621,0.265), 34.1),
    "CCCCCCCCCCCCCC":    ("tetradecane",         (1.4290,0.00,0.00,0.0259, 2.03,0.000,0.000), (-0.04,0.00,0.00,0.00),(0.000,0.013,0.700,0.000), 30.9),
    "CCCCCCCCCCCCCCCC":  ("hexadecane",          (1.4345,0.00,0.00,0.0272, 2.05,0.000,0.000), (-0.04,0.00,0.00,0.00),(0.000,0.013,0.712,0.000), 30.9),
    "CCC(C)CC":          ("3-methylpentane",     (1.3765,0.00,0.00,0.0176, 1.89,0.000,0.000), (-0.08,0.00,0.00,0.00),(0.000,0.013,0.622,0.000), 30.9),
    "COCCOC":            ("dme",                 (1.3796,0.00,0.41,0.0220, 7.20,0.000,0.000), (0.53,0.00,0.41,0.00), (0.000,0.557,0.661,0.476), 38.2),
    "CCCCCCCCCCCO":      ("1-undecanol",         (1.4392,0.37,0.48,0.0285, 7.93,0.000,0.000), (0.40,0.77,0.81,0.00), (0.287,0.810,0.726,0.410), 47.5),
    "CCCCCl":            ("1-chloropentane",     (1.4127,0.00,0.10,0.0246, 6.65,0.000,0.000), (0.39,0.00,0.10,0.00), (0.000,0.082,0.694,0.362), 36.0),
    "ClC=C(Cl)Cl":       ("tce",                 (1.4773,0.08,0.03,0.0297, 3.39,0.000,0.333), (0.53,0.08,0.03,0.00), (0.000,0.025,0.791,0.482), 39.4),
    "O=C1CCCCC1":        ("cyclohexanone",       (1.4507,0.00,0.53,0.0360,15.50,0.000,0.000), (0.76,0.00,0.53,0.00), (0.000,0.478,0.762,0.844), 39.8),
    "CCCCCCCCC":         ("nonane",              (1.4054,0.00,0.00,0.0222, 1.97,0.000,0.000), (-0.04,0.00,0.00,0.00),(0.000,0.013,0.663,0.000), 30.9),
    "OC1CCCCC1":         ("cyclohexanol",        (1.4641,0.66,0.84,0.0330,15.00,0.000,0.000), (0.44,0.66,0.84,0.00), (0.369,0.971,0.683,0.607), 47.7),
    "C1CCC2CCCCC2C1":    ("decalin",             (1.4810,0.00,0.00,0.0268, 2.15,0.000,0.000), (0.00,0.00,0.00,0.00), (0.000,0.073,0.737,0.000), 30.9),
    "COc1ccccc1":        ("anisole",             (1.5143,0.00,0.29,0.0317, 4.30,0.429,0.000), (0.73,0.00,0.29,1.00), (0.000,0.299,0.817,0.520), 37.1),
    "Brc1ccccc1":        ("bromobenzene",        (1.5597,0.00,0.06,0.0366, 5.40,0.286,0.000), (0.79,0.00,0.06,1.00), (0.000,0.095,0.882,0.497), 37.0),
    "O=C1CCCO1":         ("gbl",                 (1.4361,0.00,0.64,0.0380,39.10,0.000,0.000), (0.87,0.00,0.49,0.00), (0.000,0.429,0.757,0.961), 44.1),
    "CN(C)C(=O)N(C)C":   ("tmur",               (1.4543,0.00,0.80,0.0355,23.06,0.000,0.000), (0.83,0.00,0.80,0.00), (0.000,0.662,0.793,0.895), 41.2),
    "Clc1ccccc1Cl":      ("odcb",               (1.5514,0.00,0.03,0.0376, 9.93,0.286,0.286), (0.80,0.00,0.03,1.00), (0.000,0.069,0.872,0.593), 38.0),
    "S=C=S":             ("carbon disulfide",    (1.6241,0.00,0.07,0.0319, 2.64,0.000,0.000), (0.61,0.00,0.07,0.00), (0.000,0.031,0.899,0.221), 32.8),
    "CCCCCCCCCCCCCCCCCl": ("1-chlorohexadecane",(1.4479,0.00,0.10,0.0299, 4.45,0.000,0.000), (0.30,0.00,0.10,0.00), (0.000,0.082,0.750,0.283), 35.0),
}


def _build_table(raw, idx):
    out = {}
    for smi, row in raw.items():
        csmi = canonical(smi)
        if csmi:
            out[csmi] = row[idx]
    return out


MNSDDB  = _build_table(_RAW, 1)
KT      = _build_table(_RAW, 2)
CATALAN = _build_table(_RAW, 3)
ET30    = _build_table(_RAW, 4)

mnsddb_cols  = ["solv_n", "solv_alpha", "solv_beta", "solv_gamma", "solv_epsilon", "solv_phi", "solv_psi"]
kt_cols      = ["solv_kt_pistar", "solv_kt_alpha", "solv_kt_beta", "solv_kt_delta"]
catalan_cols = ["solv_cat_SA", "solv_cat_SB", "solv_cat_SP", "solv_cat_SdP"]

BCUT2D_NAMES = [
    "BCUT2D_MWHI", "BCUT2D_MWLOW", "BCUT2D_CHGHI", "BCUT2D_CHGLO",
    "BCUT2D_LOGPHI", "BCUT2D_LOGPLOW", "BCUT2D_MRHI", "BCUT2D_MRLOW",
]


def compute_bcut2d(smi, prefix=""):
    mol = mol_from_smiles(smi)
    if mol is None:
        return {f"{prefix}{n}": np.nan for n in BCUT2D_NAMES}
    try:
        vals = rdMolDescriptors.BCUT2D(mol)
        return {f"{prefix}{n}": v for n, v in zip(BCUT2D_NAMES, vals)}
    except Exception:
        return {f"{prefix}{n}": np.nan for n in BCUT2D_NAMES}


RDKIT_DESC_FUNCS = {
    "desc_MolWt":                   Descriptors.MolWt,
    "desc_HeavyAtomMolWt":          Descriptors.HeavyAtomMolWt,
    "desc_MolLogP":                 Crippen.MolLogP,
    "desc_MolMR":                   Crippen.MolMR,
    "desc_TPSA":                    Descriptors.TPSA,
    "desc_LabuteASA":               Descriptors.LabuteASA,
    "desc_NumHDonors":              Descriptors.NumHDonors,
    "desc_NumHAcceptors":           Descriptors.NumHAcceptors,
    "desc_NumRotatableBonds":       Descriptors.NumRotatableBonds,
    "desc_NumAromaticRings":        Descriptors.NumAromaticRings,
    "desc_NumAliphaticRings":       Descriptors.NumAliphaticRings,
    "desc_NumAromaticCarbocycles":  Descriptors.NumAromaticCarbocycles,
    "desc_NumAromaticHeterocycles": Descriptors.NumAromaticHeterocycles,
    "desc_RingCount":               Descriptors.RingCount,
    "desc_FractionCSP3":            Descriptors.FractionCSP3,
    "desc_HeavyAtomCount":          Descriptors.HeavyAtomCount,
    "desc_BertzCT":                 Descriptors.BertzCT,
    "desc_VSA_EState2":             Descriptors.VSA_EState2,
    "desc_VSA_EState3":             Descriptors.VSA_EState3,
    "desc_MaxPartialCharge":        Descriptors.MaxPartialCharge,
    "desc_MinPartialCharge":        Descriptors.MinPartialCharge,
    "desc_Chi0":                    Descriptors.Chi0,
    "desc_Chi1":                    Descriptors.Chi1,
    "desc_Kappa1":                  Descriptors.Kappa1,
    "desc_Kappa2":                  Descriptors.Kappa2,
    "desc_Kappa3":                  Descriptors.Kappa3,
    "desc_qed":                     Descriptors.qed,
}


def compute_rdkit_descs(smi):
    mol = mol_from_smiles(smi)
    if mol is None:
        return {k: np.nan for k in RDKIT_DESC_FUNCS}
    out = {}
    for name, func in RDKIT_DESC_FUNCS.items():
        try:
            out[name] = func(mol)
        except Exception:
            out[name] = np.nan
    return out


morgan_gen = GetMorganGenerator(radius=3, fpSize=1024)


def to_morgan(smi):
    mol = mol_from_smiles(smi)
    return list(morgan_gen.GetFingerprintAsNumPy(mol)) if mol else [np.nan] * 1024


def to_maccs(smi):
    mol = mol_from_smiles(smi)
    return list(MACCSkeys.GenMACCSKeys(mol)) if mol else [np.nan] * 167


def build_feature_table(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    required = ["Chromophore", "Solvent", "Absorption max (nm)", "Emission max (nm)"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {input_csv}: {missing}")

    df["Solvent"] = df["Solvent"].replace({"gas": np.nan})
    df["_solv_canon"] = df["Solvent"].apply(canonical)

    for cols, table, n in [
        (mnsddb_cols,  MNSDDB,  7),
        (kt_cols,      KT,      4),
        (catalan_cols, CATALAN, 4),
    ]:
        vals = df["_solv_canon"].apply(
            lambda x: table.get(x, (np.nan,) * n)
        ).apply(pd.Series)
        vals.columns = cols
        df = pd.concat([df.reset_index(drop=True), vals], axis=1)

    df["solv_ET30"] = df["_solv_canon"].apply(lambda x: ET30.get(x, np.nan))

    df = df[df["solv_epsilon"].notna()].reset_index(drop=True)

    df = pd.concat(
        [df, df["Chromophore"].apply(lambda s: pd.Series(compute_bcut2d(s, "chrom_")))],
        axis=1,
    ).reset_index(drop=True)

    _bcut_cache = {}

    def solv_bcut(csmi):
        if csmi not in _bcut_cache:
            _bcut_cache[csmi] = compute_bcut2d(csmi, "solv_bcut_")
        return _bcut_cache[csmi]

    df = pd.concat(
        [df, df["_solv_canon"].apply(
            lambda x: pd.Series(solv_bcut(x) if x else {f"solv_bcut_{n}": np.nan for n in BCUT2D_NAMES})
        )],
        axis=1,
    ).reset_index(drop=True)

    df = pd.concat(
        [df, df["Chromophore"].apply(lambda s: pd.Series(compute_rdkit_descs(s)))],
        axis=1,
    ).reset_index(drop=True)

    morgan_fp = df["Chromophore"].apply(to_morgan).apply(pd.Series)
    morgan_fp.columns = [f"morgan_{i}" for i in range(1024)]

    maccs_fp = df["Chromophore"].apply(to_maccs).apply(pd.Series)
    maccs_fp.columns = [f"maccs_{i}" for i in range(167)]

    meta   = ["Chromophore", "Solvent"]
    target = [c for c in ["Absorption max (nm)", "Emission max (nm)", "Stokes Shift (nm)"] if c in df.columns]

    solv_f  = (
        mnsddb_cols
        + kt_cols
        + catalan_cols
        + ["solv_ET30"]
        + [f"solv_bcut_{n}" for n in BCUT2D_NAMES]
    )
    chrom_f = [f"chrom_{n}" for n in BCUT2D_NAMES] + list(RDKIT_DESC_FUNCS.keys())

    df_out = pd.concat(
        [df[meta + target + solv_f + chrom_f].reset_index(drop=True),
         morgan_fp,
         maccs_fp],
        axis=1,
    )

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df_out.to_csv(output_csv, index=False)


if __name__ == "__main__":
    build_feature_table(
        input_csv="datasetname_processed.csv",
        output_csv="datasetname_features.csv",
    )
