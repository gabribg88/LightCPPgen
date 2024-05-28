import os
import math
import time
import pandas as pd
from functools import reduce
from collections import defaultdict

from peptides import Peptide
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio import SeqIO
from rdkit import Chem
from rdkit.Chem import Lipinski
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Crippen
from rdkit.Chem import Fragments
import iFeatureOmegaCLI

def create_comb_MLCPP2(train_fasta_file, train_targets_file, test_fasta_file, test_targets_file):
    ## create train
    data = defaultdict(list)
    with open(train_fasta_file) as fp:
        for record in SeqIO.parse(fp,"fasta"):
            idx = str(record.id)
            sequence = str(record.seq)
            data['ID'].append(idx)
            data['Sequence'].append(sequence)
    train = pd.DataFrame.from_dict(data)
    train = pd.merge(train, pd.read_csv(train_targets_file), left_index=True, right_index=True)
    train['Dataset'] = 'train'
    ## create test
    data = defaultdict(list)
    with open(test_fasta_file) as fp:
        for record in SeqIO.parse(fp,"fasta"):
            idx = str(record.id)
            sequence = str(record.seq)
            data['ID'].append(idx)
            data['Sequence'].append(sequence)
    test = pd.DataFrame.from_dict(data)
    test = pd.merge(test, pd.read_csv(test_targets_file), left_index=True, right_index=True)
    test['Dataset'] = 'test'
    ## create comb
    comb = pd.concat([train, test], ignore_index=True)
    return comb

def add_structure_based_descriptors(df, dataset_name='train', save_folder=None, return_df=False):
    df_tmp = df.copy()
    # Average Molecular Weight (MW)
    df_tmp['MW'] = df_tmp.Sequence.apply(lambda x: Descriptors.MolWt(Chem.MolFromFASTA(x)))
    # Number of Rotatable Bonds (NRB)
    df_tmp['NRB'] = df_tmp.Sequence.apply(lambda x: Lipinski.NumRotatableBonds(Chem.MolFromFASTA(x)))
    # topological Polar Surface Area (tPSA)
    df_tmp['tPSA'] = df_tmp.Sequence.apply(lambda x: Descriptors.TPSA(Chem.MolFromFASTA(x)))
    # Fraction of sp3-hybridized carbon atoms (Fsp3)
    df_tmp['Fsp3'] = df_tmp.Sequence.apply(lambda x: rdMolDescriptors.CalcFractionCSP3(Chem.MolFromFASTA(x)))
    # Octanol-water partition coefficient (cLogP)
    df_tmp['cLogP'] = df_tmp.Sequence.apply(lambda x: Crippen.MolLogP(Chem.MolFromFASTA(x)))
    # number of aromatic rings (NAR)
    df_tmp['NAR'] = df_tmp.Sequence.apply(lambda x: Lipinski.NumAromaticRings(Chem.MolFromFASTA(x)))
    # number of hydrogen bond donors (HBD)
    df_tmp['HBD'] = df_tmp.Sequence.apply(lambda x: Lipinski.NumHDonors(Chem.MolFromFASTA(x)))
    # number of hydrogen bond acceptors (HBA)
    df_tmp['HBA'] = df_tmp.Sequence.apply(lambda x: Lipinski.NumHAcceptors(Chem.MolFromFASTA(x)))
    # difference between the numbers of hydrogen bond donors and acceptors
    df_tmp['HBD_minus_HBA'] = df_tmp['HBD'] - df_tmp['HBA']
    # number of primary amino groups (NPA)
    df_tmp['NPA'] = df_tmp.Sequence.apply(lambda x: Fragments.fr_NH2(Chem.MolFromFASTA(x)))
    # number of guanidinium groups (NG)
    df_tmp['NG'] = df_tmp.Sequence.apply(lambda x: Fragments.fr_guanido(Chem.MolFromFASTA(x)))
    # net charge (NetC)
    df_tmp['NetC'] = df_tmp.Sequence.apply(lambda x: Peptide(x).charge(pH=7.4))
    # number of negatively charged amino acids (NNCAA) at pH = 7.4.
    # TODO
    # isoelectric point
    df_tmp['IsoP'] = df_tmp.Sequence.apply(lambda x: Peptide(x).isoelectric_point())
    # hydrophobicity
    df_tmp['Hydrophobicity'] = df_tmp.Sequence.apply(lambda x: Peptide(x).hydrophobicity())
    # aromaticity
    df_tmp['Aromaticity'] = df_tmp.Sequence.apply(lambda x: ProteinAnalysis(x).aromaticity())
    # Length
    df_tmp['Length'] = df_tmp.Sequence.apply(lambda x: len(x))
    if save_folder is not None:
        df_tmp.to_pickle(os.path.join(save_folder, f'{dataset_name}_physicochemical.pickle'))
    if return_df is True:
        return df_tmp

def add_sequence_based_descriptors(df, train_fasta_file, test_fasta_file, parameters_setting_file=None, descriptors=['AAC'], dataset_name='comb', save_folder=None, return_df=False):
    comb_tmp = df.copy()
    dfList = []
    for descriptor in descriptors:
        start = time.time()
        pro = iFeatureOmegaCLI.iProtein(train_fasta_file)
        if parameters_setting_file is not None:
            pro.import_parameters(parameters_setting_file)
        pro.get_descriptor(descriptor)
        train_des = pro.encodings
        train_des['Dataset'] = 'train'
        
        pro = iFeatureOmegaCLI.iProtein(test_fasta_file)
        if parameters_setting_file is not None:
            pro.import_parameters(parameters_setting_file)
        pro.get_descriptor(descriptor)
        test_des = pro.encodings
        test_des['Dataset'] = 'test'
        
        comb_des = pd.concat([train_des, test_des], ignore_index=False)
        comb_des.reset_index(inplace=True)
        comb_des.rename({'index':'ID'}, axis=1, inplace=True)
        
        comb = pd.merge(comb_tmp, comb_des, on=['ID','Dataset'])
        end = time.time()
        print(f'{descriptor}: done in {end-start:.3f} s')
        
        if save_folder is not None:
            comb.to_pickle(os.path.join(save_folder, f"{dataset_name}_{descriptor.split(' ')[0]}.pickle"))
        if return_df is True:
            dfList.append(comb)
    if return_df is True:
        return reduce(lambda df1, df2: pd.merge(df1, df2, on=['ID', 'Sequence', 'CPP', 'Dataset']), dfList)
    


