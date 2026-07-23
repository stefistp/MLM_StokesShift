import pandas as pd

deep4chem = pd.read_csv('deep4chem.csv')

required_columns = [
    'Chromophore',
    'Solvent',
    'Absorption max (nm)',
    'Emission max (nm)',
]

deep4chem_processed = deep4chem[required_columns].copy()

deep4chem_processed['Stokes Shift (nm)'] = (
    deep4chem_processed['Emission max (nm)'] -
    deep4chem_processed['Absorption max (nm)']
)

all_columns = required_columns + ['Stokes Shift (nm)']

deep4chem_processed = deep4chem_processed.dropna(subset=all_columns)
deep4chem_processed = deep4chem_processed[deep4chem_processed['Stokes Shift (nm)'] > 0]

deep4chem_processed.to_csv('deep4chem_processed.csv', index=False)
