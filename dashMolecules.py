import pandas as pd
from rdkit.Chem import PandasTools
from constants import PROJECT_PATH

if __name__ == "__main__":
    frame = PandasTools.LoadSDF(str(PROJECT_PATH/"download_eth"/'sdf_qmugs500_mbis_collect.sdf'),smilesName='SMILES',molColName='Molecule',includeFingerprints=False)
    frame["comp_key"] = frame["CONF_ID"].astype("string")+frame["DASH_IDX"]
    df = pd.read_csv("atomData.csv",usecols=['DASH_IDX','atom_idx','cnf_idx', 'element', 'mulliken', 'resp1','resp2', 'dual', 'mbis_dipole_strength'])
    non_h = df[df.element!="H"]
    non_h["comp_key"] = "conf_"+non_h["cnf_idx"].astype("string").str.zfill(2) + non_h["DASH_IDX"] 
    partner = frame[["SMILES","CHEMBL_ID","comp_key"]].drop_duplicates()
    merged = pd.merge(non_h, partner, how="inner", on="comp_key")
    merged.to_csv("dash_dataset.csv")