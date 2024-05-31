import os
import math
import time
import random
import itertools
import numpy as np
import pandas as pd
import lightgbm as lgb
from functools import reduce
from collections import defaultdict

from peptides import Peptide
from Bio import Align
from Bio.Align import substitution_matrices
from sklearn.neighbors import LocalOutlierFactor



class Featurizer():
    def __init__(self, data_folder:str):
        self.data_folder = data_folder
        self.feature_names = ['CTDC_charge.G1',
                              'CTDD_hydrophobicity_ARGP820101.1.residue50',
                              'CKSAAP_IR.gap2',
                              'CTDD_normwaalsvolume.2.residue100',
                              'DDE_LA',
                              'ASDC_AS',
                              'PAAC_Xc1.V',
                              'APAAC_Pc2.Hydrophobicity.1',
                              'DDE_LP',
                              'CKSAAP_LL.gap4',
                              'APAAC_Pc1.A',
                              'CTDD_hydrophobicity_FASG890101.2.residue50',
                              'CKSAAGP_postivecharger.uncharger.gap1',
                              'KSCTriad_g5.g5.g5.gap1',
                              'ZScale_p1.z5',
                              'QSOrder_Grantham.Xr.T',
                              'ASDC_DM',
                              'CKSAAP_TR.gap4',
                              'PAAC_Xc1.G',
                              'CKSAAGP_uncharger.aromatic.gap9'
                             ]
        
    def Count(self, seq1, seq2):
        sum = 0
        for aa in seq1:
            sum = sum + seq2.count(aa)
        return sum

    def Count1(self, aaSet, sequence):
        number = 0
        for aa in sequence:
            if aa in aaSet:
                number = number + 1
        cutoffNums = [1, math.floor(0.25 * number), math.floor(0.50 * number), math.floor(0.75 * number), number]
        cutoffNums = [i if i >= 1 else 1 for i in cutoffNums]

        code = []
        for cutoff in cutoffNums:
            myCount = 0
            for i in range(len(sequence)):
                if sequence[i] in aaSet:
                    myCount += 1
                    if myCount == cutoff:
                        code.append((i + 1) / len(sequence) * 100)
                        break
            if myCount == 0:
                code.append(0)
        return code
    
    def compute_CTDC_charge_G1(self, sequence):
        return self.Count('KR', sequence) / len(sequence)
    
    def compute_CTDD_hydrophobicity_ARGP820101_1_residue50(self, sequence):
        return self.Count1('QSTNGDE', sequence)[2]
    
    def compute_CKSAAP_IR_gap2(self, sequence, normalized=True):
        AA = 'ACDEFGHIKLMNPQRSTVWY'
        gap = 2
        pair = 'IR'
        count = 0
        sum = 0

        for index1 in range(len(sequence) - gap - 1):
            index2 = index1 + gap + 1
            if sequence[index1] == pair[0] and sequence[index2] == pair[1]:
                count += 1
            sum += 1

        if normalized and sum > 0:
            feature_value = count / sum
        else:
            feature_value = count

        return feature_value
    
    def compute_CTDD_normwaalsvolume_2_residue100(self, sequence):
        return self.Count1('NVEQIL', sequence)[4]
    
    def compute_DDE_LA(self, sequence):
        AA = 'ACDEFGHIKLMNPQRSTVWY'
        myCodons = {'A': 4, 'C': 2, 'D': 2, 'E': 2, 'F': 2, 'G': 4, 'H': 2, 'I': 3, 'K': 2, 'L': 6,
                    'M': 1, 'N': 2, 'P': 4, 'Q': 2, 'R': 6, 'S': 6, 'T': 4, 'V': 4, 'W': 1, 'Y': 2
                    }
        N_LA = 0
        for i in range(len(sequence)-1):
            if sequence[i:i+2] == 'LA':
                N_LA += 1
        Dc = N_LA / (len(sequence) - 1)
        Tm = (myCodons['L'] / 61) * (myCodons['A'] / 61)
        Tv = (Tm * (1 - Tm)) / (len(sequence) - 1)
        return (Dc - Tm) / math.sqrt(Tv)
    
    def compute_ASDC_AS(self, sequence):
        AA = 'ACDEFGHIKLMNPQRSTVWY'
        AS_counter = 0
        sum = 0
        for j in range(len(sequence)):
            for k in range(j + 1, len(sequence)):
                if sequence[j] in AA and sequence[k] in AA:
                    if sequence[j] + sequence[k] == 'AS':
                        AS_counter += 1
                sum += 1
        return AS_counter / sum
    
    def compute_PAAC_Xc1_V(self, sequence):
        
        def Rvalue(aa1, aa2, AADict, Matrix):
            return sum([(Matrix[i][AADict[aa1]] - Matrix[i][AADict[aa2]]) ** 2 for i in range(len(Matrix))]) / len(Matrix)
        
        lambdaValue = 1
        w = 0.05
        dataFile = os.path.join(self.data_folder, 'PAAC.txt')
        with open(dataFile) as f:
            records = f.readlines()
        AA = ''.join(records[0].rstrip().split()[1:])
        AADict = {}
        for i in range(len(AA)):
            AADict[AA[i]] = i
        AAProperty = []
        AAPropertyNames = []
        for i in range(1, len(records)):
            array = records[i].rstrip().split() if records[i].rstrip() != '' else None
            AAProperty.append([float(j) for j in array[1:]])
            AAPropertyNames.append(array[0])
        AAProperty1 = []
        for i in AAProperty:
            meanI = sum(i) / 20
            fenmu = math.sqrt(sum([(j - meanI) ** 2 for j in i]) / 20)
            AAProperty1.append([(j - meanI) / fenmu for j in i])

        theta = []
        for n in range(1, lambdaValue + 1):
            theta.append(
                sum([Rvalue(sequence[j], sequence[j + n], AADict, AAProperty1) for j in
                    range(len(sequence) - n)]) / (
                        len(sequence) - n))

        return sequence.count('V') / (1 + w * sum(theta))
    
    def compute_APAAC_Pc2_Hydrophobicity_1(self, sequence):
        lambdaValue = 1
        w = 0.05           
        dataFile = os.path.join(self.data_folder, 'PAAC.txt')
        with open(dataFile) as f:
            records = f.readlines()
        AA = ''.join(records[0].rstrip().split()[1:])
        AADict = {}
        for i in range(len(AA)):
            AADict[AA[i]] = i
        AAProperty = []
        AAPropertyNames = []
        for i in range(1, len(records) - 1):
            array = records[i].rstrip().split() if records[i].rstrip() != '' else None
            AAProperty.append([float(j) for j in array[1:]])
            AAPropertyNames.append(array[0])
        AAProperty1 = []
        for i in AAProperty:
            meanI = sum(i) / 20
            fenmu = math.sqrt(sum([(j - meanI) ** 2 for j in i]) / 20)
            AAProperty1.append([(j - meanI) / fenmu for j in i])

        theta = []
        for n in range(1, lambdaValue + 1):
            for j in range(len(AAProperty1)):
                theta.append(
                    sum([AAProperty1[j][AADict[sequence[k]]] * AAProperty1[j][AADict[sequence[k + n]]] for k in
                        range(len(sequence) - n)]) / (len(sequence) - n))

        return w * theta[0] / (1 + w * sum(theta))
    
    def compute_DDE_LP(self, sequence):
        AA = 'ACDEFGHIKLMNPQRSTVWY'
        myCodons = {'A': 4, 'C': 2, 'D': 2, 'E': 2, 'F': 2, 'G': 4, 'H': 2, 'I': 3, 'K': 2, 'L': 6,
                    'M': 1, 'N': 2, 'P': 4, 'Q': 2, 'R': 6, 'S': 6, 'T': 4, 'V': 4, 'W': 1, 'Y': 2
                    }
        N_SP = 0
        for i in range(len(sequence)-1):
            if sequence[i:i+2] == 'LP':
                N_SP += 1
        Dc = N_SP / (len(sequence) - 1)
        Tm = (myCodons['L'] / 61) * (myCodons['P'] / 61)
        Tv = (Tm * (1 - Tm)) / (len(sequence) - 1)
        return (Dc - Tm) / math.sqrt(Tv)
    
    def compute_CKSAAP_LL_gap4(self, sequence, normalized=True):
        AA = 'ACDEFGHIKLMNPQRSTVWY'
        gap = 4
        pair = 'LL'
        count = 0
        sum = 0

        for index1 in range(len(sequence) - gap - 1):
            index2 = index1 + gap + 1
            if sequence[index1] == pair[0] and sequence[index2] == pair[1]:
                count += 1
            sum += 1

        if normalized and sum > 0:
            feature_value = count / sum
        else:
            feature_value = count

        return feature_value
    
    def compute_APAAC_Pc1_A(self, sequence):
        lambdaValue = 1
        w = 0.05
        dataFile = os.path.join(self.data_folder, 'PAAC.txt')
        with open(dataFile) as f:
            records = f.readlines()
        AA = ''.join(records[0].rstrip().split()[1:])
        AADict = {}
        for i in range(len(AA)):
            AADict[AA[i]] = i
        AAProperty = []
        AAPropertyNames = []
        for i in range(1, len(records) - 1):
            array = records[i].rstrip().split() if records[i].rstrip() != '' else None
            AAProperty.append([float(j) for j in array[1:]])
            AAPropertyNames.append(array[0])
        AAProperty1 = []
        for i in AAProperty:
            meanI = sum(i) / 20
            fenmu = math.sqrt(sum([(j - meanI) ** 2 for j in i]) / 20)
            AAProperty1.append([(j - meanI) / fenmu for j in i])
        theta = []
        for n in range(1, lambdaValue + 1):
            for j in range(len(AAProperty1)):
                theta.append(
                    sum([AAProperty1[j][AADict[sequence[k]]] * AAProperty1[j][AADict[sequence[k + n]]] for k in
                        range(len(sequence) - n)]) / (len(sequence) - n))
        return sequence.count('A') / (1 + w * sum(theta))
    
    def compute_CTDD_hydrophobicity_FASG890101_2_residue50(self, sequence):
        return self.Count1('NTPG', sequence)[2]
    
    def compute_CKSAAGP_postivecharger_uncharger_gap1(self, sequence):
        def generateGroupPairs(groupKey):
            gPair = {}
            for key1 in groupKey:
                for key2 in groupKey:
                    gPair[key1 + '.' + key2] = 0
            return gPair

        gap = 1
        group = {
            'alphaticr': 'GAVLMI',
            'aromatic': 'FYW',
            'postivecharger': 'KRH',
            'negativecharger': 'DE',
            'uncharger': 'STCPNQ'
        }
        AA = 'ARNDCQEGHILKMFPSTWYV'
        groupKey = group.keys()
        index = {}
        for key in groupKey:
            for aa in group[key]:
                index[aa] = key
        gPairIndex = []
        for key1 in groupKey:
            for key2 in groupKey:
                gPairIndex.append(key1 + '.' + key2)

        for g in range(gap + 1):
            gPair = generateGroupPairs(groupKey)
            sum = 0
            for p1 in range(len(sequence)):
                p2 = p1 + g + 1
                if p2 < len(sequence) and sequence[p1] in AA and sequence[p2] in AA:
                    gPair[index[sequence[p1]] + '.' + index[sequence[p2]]] = gPair[
                                                                                index[sequence[p1]] + '.' + index[
                                                                                    sequence[p2]]] + 1
                    sum = sum + 1
        return gPair['postivecharger.uncharger'] / sum if sum > 0 else 0
    
    def compute_KSCTriad_g5_g5_g5_gap1(self, sequence):
        def CalculateKSCTriad(sequence, gap, features, AADict):
            res = []
            for g in range(gap + 1):
                myDict = {}
                for f in features:
                    myDict[f] = 0

                for i in range(len(sequence)):
                    if i + g + 1 < len(sequence) and i + 2 * g + 2 < len(sequence):
                        fea = AADict[sequence[i]] + '.' + AADict[sequence[i + g + 1]] + '.' + AADict[
                            sequence[i + 2 * g + 2]]
                        myDict[fea] = myDict[fea] + 1

                maxValue, minValue = np.max(list(myDict.values())), np.min(list(myDict.values()))            
            return (myDict['g5.g5.g5'] - minValue) / maxValue

        gap = 1
        AAGroup = {
            'g1': 'AGV',
            'g2': 'ILFP',
            'g3': 'YMTS',
            'g4': 'HNQW',
            'g5': 'RK',
            'g6': 'DE',
            'g7': 'C'
        }
        myGroups = sorted(AAGroup.keys())
        AADict = {}
        for g in myGroups:
            for aa in AAGroup[g]:
                AADict[aa] = g
        features = [f1 + '.' + f2 + '.' + f3 for f1 in myGroups for f2 in myGroups for f3 in myGroups]
        return CalculateKSCTriad(sequence, gap, features, AADict)
    
    def compute_ZScale_p1_z5(self, sequence):
        sequence = sequence[0] + sequence[-5:]
        zscale = {
            'A': [0.24, -2.32, 0.60, -0.14, 1.30],  # A
            'C': [0.84, -1.67, 3.71, 0.18, -2.65],  # C
            'D': [3.98, 0.93, 1.93, -2.46, 0.75],  # D
            'E': [3.11, 0.26, -0.11, -0.34, -0.25],  # E
            'F': [-4.22, 1.94, 1.06, 0.54, -0.62],  # F
            'G': [2.05, -4.06, 0.36, -0.82, -0.38],  # G
            'H': [2.47, 1.95, 0.26, 3.90, 0.09],  # H
            'I': [-3.89, -1.73, -1.71, -0.84, 0.26],  # I
            'K': [2.29, 0.89, -2.49, 1.49, 0.31],  # K
            'L': [-4.28, -1.30, -1.49, -0.72, 0.84],  # L
            'M': [-2.85, -0.22, 0.47, 1.94, -0.98],  # M
            'N': [3.05, 1.62, 1.04, -1.15, 1.61],  # N
            'P': [-1.66, 0.27, 1.84, 0.70, 2.00],  # P
            'Q': [1.75, 0.50, -1.44, -1.34, 0.66],  # Q
            'R': [3.52, 2.50, -3.50, 1.99, -0.17],  # R
            'S': [2.39, -1.07, 1.15, -1.39, 0.67],  # S
            'T': [0.75, -2.18, -1.12, -1.46, -0.40],  # T
            'V': [-2.59, -2.64, -1.54, -0.85, -0.02],  # V
            'W': [-4.36, 3.94, 0.59, 3.44, -1.59],  # W
            'Y': [-2.54, 2.44, 0.43, 0.04, -1.47],  # Y
            '-': [0.00, 0.00, 0.00, 0.00, 0.00],  # -
        }
        return zscale[sequence[0]][4]
    
    def compute_QSOrder_Grantham_Xr_T(self, sequence):
        nlag = 3
        w = 0.05       
        dataFile1 = os.path.join(os.path.join('..','data', 'Grantham.txt'))
        AA = 'ACDEFGHIKLMNPQRSTVWY'
        AA1 = 'ARNDCQEGHILKMFPSTWYV'
        DictAA = {}
        for i in range(len(AA)):
            DictAA[AA[i]] = i
        DictAA1 = {}
        for i in range(len(AA1)):
            DictAA1[AA1[i]] = i

        with open(dataFile1) as f:
            records = f.readlines()[1:]
        AADistance1 = []
        for i in records:
            array = i.rstrip().split()[1:] if i.rstrip() != '' else None
            AADistance1.append(array)

        AADistance1 = np.array([float(AADistance1[i][j]) for i in range(len(AADistance1)) for j in range(len(AADistance1[i]))]).reshape((20, 20))

        arrayGM = []
        for n in range(1, nlag + 1):
            arrayGM.append(sum(
                [AADistance1[DictAA1[sequence[j]]][DictAA1[sequence[j + n]]] ** 2 for j in range(len(sequence) - n)]))

        return sequence.count('T') / (1 + w * sum(arrayGM))
    
    def compute_ASDC_DM(self, sequence):
        AA = 'ACDEFGHIKLMNPQRSTVWY'
        DM_counter = 0
        sum = 0
        for j in range(len(sequence)):
            for k in range(j + 1, len(sequence)):
                if sequence[j] in AA and sequence[k] in AA:
                    if sequence[j] + sequence[k] == 'DM':
                        DM_counter += 1
                sum += 1
        return DM_counter / sum
    
    def compute_CKSAAP_TR_gap4(self, sequence, normalized=True):
        AA = 'ACDEFGHIKLMNPQRSTVWY'
        gap = 4
        pair = 'TR'
        count = 0
        sum = 0

        for index1 in range(len(sequence) - gap - 1):
            index2 = index1 + gap + 1
            if sequence[index1] == pair[0] and sequence[index2] == pair[1]:
                count += 1
            sum += 1

        if normalized and sum > 0:
            feature_value = count / sum
        else:
            feature_value = count

        return feature_value
    
    def compute_PAAC_Xc1_G(self, sequence):
        def Rvalue(aa1, aa2, AADict, Matrix):
            return sum([(Matrix[i][AADict[aa1]] - Matrix[i][AADict[aa2]]) ** 2 for i in range(len(Matrix))]) / len(Matrix)

        lambdaValue = 1
        w = 0.05
        dataFile = os.path.join(self.data_folder, 'PAAC.txt')
        with open(dataFile) as f:
            records = f.readlines()
        AA = ''.join(records[0].rstrip().split()[1:])
        AADict = {}
        for i in range(len(AA)):
            AADict[AA[i]] = i
        AAProperty = []
        AAPropertyNames = []
        for i in range(1, len(records)):
            array = records[i].rstrip().split() if records[i].rstrip() != '' else None
            AAProperty.append([float(j) for j in array[1:]])
            AAPropertyNames.append(array[0])
        AAProperty1 = []
        for i in AAProperty:
            meanI = sum(i) / 20
            fenmu = math.sqrt(sum([(j - meanI) ** 2 for j in i]) / 20)
            AAProperty1.append([(j - meanI) / fenmu for j in i])

        theta = []
        for n in range(1, lambdaValue + 1):
            theta.append(
                sum([Rvalue(sequence[j], sequence[j + n], AADict, AAProperty1) for j in
                    range(len(sequence) - n)]) / (
                        len(sequence) - n))

        return sequence.count('G') / (1 + w * sum(theta))
    
    def compute_CKSAAGP_uncharger_aromatic_gap9(self, sequence):
        def generateGroupPairs(groupKey):
            gPair = {}
            for key1 in groupKey:
                for key2 in groupKey:
                    gPair[key1 + '.' + key2] = 0
            return gPair

        gap = 9
        group = {
            'alphaticr': 'GAVLMI',
            'aromatic': 'FYW',
            'postivecharger': 'KRH',
            'negativecharger': 'DE',
            'uncharger': 'STCPNQ'
        }
        AA = 'ARNDCQEGHILKMFPSTWYV'
        groupKey = group.keys()
        index = {}
        for key in groupKey:
            for aa in group[key]:
                index[aa] = key
        gPairIndex = []
        for key1 in groupKey:
            for key2 in groupKey:
                gPairIndex.append(key1 + '.' + key2)

        for g in range(gap + 1):
            gPair = generateGroupPairs(groupKey)
            sum = 0
            for p1 in range(len(sequence)):
                p2 = p1 + g + 1
                if p2 < len(sequence) and sequence[p1] in AA and sequence[p2] in AA:
                    gPair[index[sequence[p1]] + '.' + index[sequence[p2]]] = gPair[
                                                                                index[sequence[p1]] + '.' + index[
                                                                                    sequence[p2]]] + 1
                    sum = sum + 1
        return gPair['uncharger.aromatic'] / sum if sum > 0 else 0
        
    def compute_features(self, sequence):
        features = pd.DataFrame(data=0.0, index=range(1), columns=self.feature_names)
        features['CTDC_charge.G1'] = self.compute_CTDC_charge_G1(sequence)
        features['CTDD_hydrophobicity_ARGP820101.1.residue50'] = self.compute_CTDD_hydrophobicity_ARGP820101_1_residue50(sequence)
        features['CKSAAP_IR.gap2'] = self.compute_CKSAAP_IR_gap2(sequence)
        features['CTDD_normwaalsvolume.2.residue100'] = self.compute_CTDD_normwaalsvolume_2_residue100(sequence)
        features['DDE_LA'] = self.compute_DDE_LA(sequence)
        features['ASDC_AS'] = self.compute_ASDC_AS(sequence)
        features['PAAC_Xc1.V'] = self.compute_PAAC_Xc1_V(sequence)
        features['APAAC_Pc2.Hydrophobicity.1'] = self.compute_APAAC_Pc2_Hydrophobicity_1(sequence)
        features['DDE_LP'] = self.compute_DDE_LP(sequence)
        features['CKSAAP_LL.gap4'] = self.compute_CKSAAP_LL_gap4(sequence)
        features['APAAC_Pc1.A'] = self.compute_APAAC_Pc1_A(sequence)
        features['CTDD_hydrophobicity_FASG890101.2.residue50'] = self.compute_CTDD_hydrophobicity_FASG890101_2_residue50(sequence)
        features['CKSAAGP_postivecharger.uncharger.gap1'] = self.compute_CKSAAGP_postivecharger_uncharger_gap1(sequence)
        features['KSCTriad_g5.g5.g5.gap1'] = self.compute_KSCTriad_g5_g5_g5_gap1(sequence)
        features['ZScale_p1.z5'] = self.compute_ZScale_p1_z5(sequence)
        features['QSOrder_Grantham.Xr.T'] = self.compute_QSOrder_Grantham_Xr_T(sequence)
        features['ASDC_DM'] = self.compute_ASDC_DM(sequence)
        features['CKSAAP_TR.gap4'] = self.compute_CKSAAP_TR_gap4(sequence)
        features['PAAC_Xc1.G'] = self.compute_PAAC_Xc1_G(sequence)
        features['CKSAAGP_uncharger.aromatic.gap9'] = self.compute_CKSAAGP_uncharger_aromatic_gap9(sequence)

        
        return features



class Featurizer_old():
    def __init__(self, data_folder:str):
        self.data_folder = data_folder
        self.feature_names = ['CTDC_charge.G1',
                             'CTDD_hydrophobicity_ARGP820101.1.residue50',
                             'SOCNumber_gGrantham.lag1',
                             'DDE_LA',
                             'Geary_ANDN920101.lag2',
                             'DDE_SP',
                             'NetC',
                             'ASDC_VG',
                             'CTDC_solventaccess.G3',
                             'APAAC_Pc1.V']
        
    def Count(self, seq1, seq2):
        sum = 0
        for aa in seq1:
            sum = sum + seq2.count(aa)
        return sum

    def Count1(self, aaSet, sequence):
        number = 0
        for aa in sequence:
            if aa in aaSet:
                number = number + 1
        cutoffNums = [1, math.floor(0.25 * number), math.floor(0.50 * number), math.floor(0.75 * number), number]
        cutoffNums = [i if i >= 1 else 1 for i in cutoffNums]

        code = []
        for cutoff in cutoffNums:
            myCount = 0
            for i in range(len(sequence)):
                if sequence[i] in aaSet:
                    myCount += 1
                    if myCount == cutoff:
                        code.append((i + 1) / len(sequence) * 100)
                        break
            if myCount == 0:
                code.append(0)
        return code
    
    def compute_CTDC_charge_G1(self, sequence):
        return self.Count('KR', sequence) / len(sequence)
    
    def compute_CTDD_hydrophobicity_ARGP820101_1_residue50(self, sequence):
        return self.Count1('QSTNGDE', sequence)[2]
    
    def compute_SOCNumber_gGrantham_lag1(self, sequence):
        n=1
        dataFile1 = os.path.join(self.data_folder, 'Grantham.txt')
        AA1 = 'ARNDCQEGHILKMFPSTWYV'
        DictAA1 = {}
        for i in range(len(AA1)):
            DictAA1[AA1[i]] = i
        with open(dataFile1) as f:
            records = f.readlines()[1:]
        AADistance1 = []
        for i in records:
            array = i.rstrip().split()[1:] if i.rstrip() != '' else None
            AADistance1.append(array)
        AADistance1 = np.array([float(AADistance1[i][j]) for i in range(len(AADistance1)) for j in range(len(AADistance1[i]))]).reshape((20, 20))
        return sum([AADistance1[DictAA1[sequence[j]]][DictAA1[sequence[j + n]]] ** 2 for j in range(len(sequence) - n)]) / (len(sequence) - n)
    
    def compute_DDE_LA(self, sequence):
        AA = 'ACDEFGHIKLMNPQRSTVWY'
        myCodons = {'A': 4, 'C': 2, 'D': 2, 'E': 2, 'F': 2, 'G': 4, 'H': 2, 'I': 3, 'K': 2, 'L': 6,
                    'M': 1, 'N': 2, 'P': 4, 'Q': 2, 'R': 6, 'S': 6, 'T': 4, 'V': 4, 'W': 1, 'Y': 2
                    }
        N_LA = 0
        for i in range(len(sequence)-1):
            if sequence[i:i+2] == 'LA':
                N_LA += 1
        Dc = N_LA / (len(sequence) - 1)
        Tm = (myCodons['L'] / 61) * (myCodons['A'] / 61)
        Tv = (Tm * (1 - Tm)) / (len(sequence) - 1)
        return (Dc - Tm) / math.sqrt(Tv)
    
    def compute_Geary_ANDN920101_lag2(self, sequence):
        props = 'ANDN920101'.split(';')
        nlag = 2
        fileAAidx = os.path.join(self.data_folder, 'AAidx.txt')
        AA = 'ARNDCQEGHILKMFPSTWYV'
        with open(fileAAidx) as f:
            records = f.readlines()[1:]
        myDict = {}
        for i in records:
            array = i.rstrip().split('\t')
            myDict[array[0]] = array[1:]
        AAidx = []
        AAidxName = []
        for i in props:
            if i in myDict:
                AAidx.append(myDict[i])
                AAidxName.append(i)
            else:
                print('"' + i + '" properties not exist.')
        AAidx1 = np.array([float(j) for i in AAidx for j in i])
        AAidx = AAidx1.reshape((len(AAidx), 20))
        propMean = np.mean(AAidx, axis=1)
        propStd = np.std(AAidx, axis=1)
        for i in range(len(AAidx)):
            for j in range(len(AAidx[i])):
                AAidx[i][j] = (AAidx[i][j] - propMean[i]) / propStd[i]
        index = {}
        for i in range(len(AA)):
            index[AA[i]] = i

        N = len(sequence)
        for prop in range(len(props)):
            xmean = sum([AAidx[prop][index[aa]] for aa in sequence]) / N
            for n in range(1, nlag + 1):
                if len(sequence) > nlag:
                    # if key is '-', then the value is 0
                    rn = (N - 1) / (2 * (N - n)) * ((sum(
                        [(AAidx[prop][index.get(sequence[j], 0)] - AAidx[prop][index.get(sequence[j + n], 0)]) ** 2
                        for
                        j in range(len(sequence) - n)])) / (sum(
                        [(AAidx[prop][index.get(sequence[j], 0)] - xmean) ** 2 for j in range(len(sequence))])))
                else:
                    rn = 'NA'
        return rn
    
    def compute_DDE_SP(self, sequence):
        AA = 'ACDEFGHIKLMNPQRSTVWY'
        myCodons = {'A': 4, 'C': 2, 'D': 2, 'E': 2, 'F': 2, 'G': 4, 'H': 2, 'I': 3, 'K': 2, 'L': 6,
                    'M': 1, 'N': 2, 'P': 4, 'Q': 2, 'R': 6, 'S': 6, 'T': 4, 'V': 4, 'W': 1, 'Y': 2
                    }
        N_SP = 0
        for i in range(len(sequence)-1):
            if sequence[i:i+2] == 'SP':
                N_SP += 1
        Dc = N_SP / (len(sequence) - 1)
        Tm = (myCodons['S'] / 61) * (myCodons['P'] / 61)
        Tv = (Tm * (1 - Tm)) / (len(sequence) - 1)
        return (Dc - Tm) / math.sqrt(Tv)
    
    def compute_NetC(self, sequence):
        return Peptide(sequence).charge(pH=7.4)
    
    def compute_ASDC_VG(self, sequence):
        AA = 'ACDEFGHIKLMNPQRSTVWY'
        VG_counter = 0
        sum = 0
        for j in range(len(sequence)):
            for k in range(j + 1, len(sequence)):
                if sequence[j] in AA and sequence[k] in AA:
                    if sequence[j] + sequence[k] == 'VG':
                        VG_counter += 1
                sum += 1
        return VG_counter / sum
    
    def compute_CTDC_solventaccess_G3(self, sequence):
        return 1 - (self.Count('ALFCGIVW', sequence) / len(sequence)) - (self.Count('RKQEND', sequence) / len(sequence))
    
    def compute_APAAC_Pc1_V(self, sequence):
        lambdaValue = 1
        w = 0.05
        dataFile = os.path.join(self.data_folder, 'PAAC.txt')
        with open(dataFile) as f:
            records = f.readlines()
        AA = ''.join(records[0].rstrip().split()[1:])
        AADict = {}
        for i in range(len(AA)):
            AADict[AA[i]] = i
        AAProperty = []
        AAPropertyNames = []
        for i in range(1, len(records) - 1):
            array = records[i].rstrip().split() if records[i].rstrip() != '' else None
            AAProperty.append([float(j) for j in array[1:]])
            AAPropertyNames.append(array[0])
        AAProperty1 = []
        for i in AAProperty:
            meanI = sum(i) / 20
            fenmu = math.sqrt(sum([(j - meanI) ** 2 for j in i]) / 20)
            AAProperty1.append([(j - meanI) / fenmu for j in i])
        theta = []
        for n in range(1, lambdaValue + 1):
            for j in range(len(AAProperty1)):
                theta.append(
                    sum([AAProperty1[j][AADict[sequence[k]]] * AAProperty1[j][AADict[sequence[k + n]]] for k in
                        range(len(sequence) - n)]) / (len(sequence) - n))
        return sequence.count('V') / (1 + w * sum(theta))
        
    def compute_features(self, sequence):
        features = pd.DataFrame(data=0.0, index=range(1), columns=self.feature_names)
        features['CTDC_charge.G1'] = self.compute_CTDC_charge_G1(sequence)
        features['CTDD_hydrophobicity_ARGP820101.1.residue50'] = self.compute_CTDD_hydrophobicity_ARGP820101_1_residue50(sequence)
        features['SOCNumber_gGrantham.lag1'] = self.compute_SOCNumber_gGrantham_lag1(sequence)
        features['DDE_LA'] = self.compute_DDE_LA(sequence)
        features['Geary_ANDN920101.lag2'] = self.compute_Geary_ANDN920101_lag2(sequence)
        features['DDE_SP'] = self.compute_DDE_SP(sequence)
        features['NetC'] = self.compute_NetC(sequence)
        features['ASDC_VG'] = self.compute_ASDC_VG(sequence)
        features['CTDC_solventaccess.G3'] = self.compute_CTDC_solventaccess_G3(sequence)
        features['APAAC_Pc1.V'] = self.compute_APAAC_Pc1_V(sequence)
        return features
    
    
# Function to predict penetration
def pred_penetration(features_query, models, best_iteration, models_number):
    proba_p = 0.0
    if isinstance(models, lgb.basic.Booster):
        proba_p = models.predict(features_query, num_iteration=best_iteration)
    elif isinstance(models, lgb.engine.CVBooster):
        for model in models:
            proba_p += model.predict(features_query, num_iteration=best_iteration) / models_number
    else:
        print('Type of model not supported')
        return 0
    return proba_p[0]

# Function to calculate anomaly score
def anomaly_score(features_query, clf_anomaly):
    pos_score = -clf_anomaly.score_samples(features_query)
    return pos_score

# Function to compute distance
def compute_distance(aligner, TARGET, query):
    weight = aligner.align(TARGET, TARGET)
    alignments = aligner.align(TARGET, query)
    distance = 1 - alignments.score / weight.score
    return distance

# Function to compute fitness score
def my_fitness(query, params_fit, weights={'w1':1, 'w2':1, 'w3':1}):
    models = params_fit['model']
    featurizer = params_fit['featurizer']
    best_iteration = params_fit['best_iteration']
    n_models = params_fit['n_models']
    w1 = weights['w1']
    w2 = weights['w2']
    w3 = weights['w3']
    
    features_query = featurizer.compute_features(query)
    proba_p = pred_penetration(features_query, models, best_iteration, n_models)

    clf_anomaly = params_fit['clf_anomaly']
    pos_score = anomaly_score(features_query, clf_anomaly)[0]

    TARGET = params_fit['target_ligand']
    aligner = params_fit['aligner']
    distance = compute_distance(aligner, TARGET, query)

    num_diff_residues = sum(1 for q, t in zip(query, TARGET) if q != t)

    a = 1
    b = 1
    x_s = 1.5
    anom_scor = np.max((b*(pos_score**2-x_s**2), 0))
    fitness = w1*distance  + w2*np.max( ( a*(1 - proba_p)**2-a*(0.2)**2, 0)) + w3*anom_scor
    return fitness, num_diff_residues

# Class defining an Individual
class Individual():
    def __init__(self, chromosome, params_fit={}):
        self.chromosome = chromosome
        self.params_fit = params_fit
        self.fitness, self.num_diff_residues = self.comp_fitness(params_fit)

    # Compute fitness according to my_fitness function
    def comp_fitness(self, params_fit):
        my_fun = params_fit['obj']
        fitness, num_diff_residues = my_fun(''.join(self.chromosome), params_fit)
        return fitness, num_diff_residues

    # Update the chromosome of the individual
    def update_chromosome(self, chromosome):
        self.chromosome = chromosome

    # Method to create random genes
    @staticmethod
    def mutated_genes(genes):
        gene = random.choice(genes)
        return gene

        # Method to create chromosome or string of genes

    @staticmethod
    def create_gnome(len_gnome, genes, target, max_diff_pct):
        num_diff = int(len_gnome * max_diff_pct / 100)  # calculate the number of different letters allowed
        gnome = list(target)  # start from the target sequence
        for _ in range(num_diff):
            idx = random.randint(0, len_gnome - 1)  # select a random position
            new_gene = Individual.mutated_genes(genes)  # generate a new gene
            while new_gene == gnome[idx]:  # ensure the new gene is different from the existing one
                new_gene = Individual.mutated_genes(genes)
            gnome[idx] = new_gene  # replace the gene at the selected position
        return gnome
        # Perform mating and produce new offspring
    def mate(self, par2, genes):
            child_chromosome = []
            for gp1, gp2 in zip(self.chromosome, par2.chromosome):
                prob = random.random()

                if prob < 0.45:
                    child_chromosome.append(gp1)
                elif prob < 0.90:
                    child_chromosome.append(gp2)
                else:
                    child_chromosome.append(Individual.mutated_genes(genes))
            return Individual(child_chromosome, self.params_fit)