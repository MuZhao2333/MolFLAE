import json
import os
from rdkit import Chem, Geometry
from rdkit.Chem import AllChem
import numpy as np
from openbabel import openbabel as ob
from scipy.spatial.distance import pdist, squareform
import itertools
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import rdFMCS
class MoleculeBuilder:
    def __init__(self):
        pass

    def make_obmol(self, xyz, atomic_numbers):
        mol = ob.OBMol()
        mol.BeginModify()
        atoms = []
        if isinstance(xyz, np.ndarray):
            xyz = xyz.tolist()
        if isinstance(atomic_numbers, np.ndarray):
            atomic_numbers = atomic_numbers.tolist()

        for xyz, t in zip(xyz, atomic_numbers):
            x, y, z = xyz
            # print(type(xyz))
            # print(type(x))
            if not (isinstance(x, (int, float)) and isinstance(y, (int, float)) and isinstance(z, (int, float))):
                raise ValueError(f"Invalid coordinates: {xyz}")
            atom = mol.NewAtom()
            atom.SetAtomicNum(t)
            atom.SetVector(x, y, z)
            atoms.append(atom)
        return mol, atoms

    def connect_the_dots(self, mol, atoms, indicators=None, covalent_factor=1.3):
        pt = Chem.GetPeriodicTable()
        if len(atoms) == 0:
            return
        mol.BeginModify()
        coords = np.array([(a.GetX(), a.GetY(), a.GetZ()) for a in atoms])
        dists = squareform(pdist(coords))
        for i, j in itertools.combinations(range(len(atoms)), 2):
            a = atoms[i]
            b = atoms[j]
            a_r = ob.GetCovalentRad(a.GetAtomicNum()) * covalent_factor
            b_r = ob.GetCovalentRad(b.GetAtomicNum()) * covalent_factor
            if dists[i, j] < a_r + b_r:
                flag = 0
                if indicators and indicators[i] and indicators[j]:
                    flag = ob.OB_AROMATIC_BOND
                mol.AddBond(a.GetIdx(), b.GetIdx(), 1, flag)
        atom_maxb = {}
        for i, a in enumerate(atoms):
            maxb = min(ob.GetMaxBonds(a.GetAtomicNum()), pt.GetDefaultValence(a.GetAtomicNum()))
            if a.GetAtomicNum() == 16 and self.count_nbrs_of_elem(a, 8) >= 2:
                maxb = 6
            atom_maxb[a.GetIdx()] = maxb
        for bond in ob.OBMolBondIter(mol):
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            if atom_maxb[a1.GetIdx()] == 1 and atom_maxb[a2.GetIdx()] == 1:
                mol.DeleteBond(bond)
        def get_bond_info(biter):
            bonds = [b for b in biter]
            binfo = []
            for bond in bonds:
                bdist = bond.GetLength()
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                ideal = ob.GetCovalentRad(a1.GetAtomicNum()) + ob.GetCovalentRad(a2.GetAtomicNum())
                stretch = bdist / ideal
                binfo.append((stretch, bond))
            binfo.sort(reverse=True, key=lambda t: t[0])
            return binfo
        binfo = get_bond_info(ob.OBMolBondIter(mol))
        for stretch, bond in binfo:
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            if stretch > 1.2 or self.forms_small_angle(a1, a2) or self.forms_small_angle(a2, a1):
                if not self.reachable(a1, a2):
                    continue
                mol.DeleteBond(bond)
        hypers = [(atom_maxb[a.GetIdx()], a.GetExplicitValence() - atom_maxb[a.GetIdx()], a) for a in atoms]
        hypers = sorted(hypers, key=lambda aa: (aa[0], -aa[1]))
        for mb, diff, a in hypers:
            if a.GetExplicitValence() <= atom_maxb[a.GetIdx()]:
                continue
            binfo = get_bond_info(ob.OBAtomBondIter(a))
            for stretch, bond in binfo:
                if stretch < 0.9:
                    continue
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                if a1.GetExplicitValence() > atom_maxb[a1.GetIdx()] or a2.GetExplicitValence() > atom_maxb[a2.GetIdx()]:
                    if not self.reachable(a1, a2):
                        continue
                    mol.DeleteBond(bond)
                    if a.GetExplicitValence() <= atom_maxb[a.GetIdx()]:
                        break
        mol.EndModify()

    def convert_ob_mol_to_rd_mol(self, ob_mol):
        ob_mol.DeleteHydrogens()
        n_atoms = ob_mol.NumAtoms()
        rd_mol = Chem.RWMol()
        rd_conf = Chem.Conformer(n_atoms)
        for ob_atom in ob.OBMolAtomIter(ob_mol):
            rd_atom = Chem.Atom(ob_atom.GetAtomicNum())
            if ob_atom.IsAromatic() and ob_atom.IsInRing() and ob_atom.MemberOfRingSize() <= 6:
                rd_atom.SetIsAromatic(True)
            i = rd_mol.AddAtom(rd_atom)
            ob_coords = ob_atom.GetVector()
            x = ob_coords.GetX()
            y = ob_coords.GetY()
            z = ob_coords.GetZ()
            rd_conf.SetAtomPosition(i, Geometry.Point3D(x, y, z))
        rd_mol.AddConformer(rd_conf)
        for ob_bond in ob.OBMolBondIter(ob_mol):
            i = ob_bond.GetBeginAtomIdx() - 1
            j = ob_bond.GetEndAtomIdx() - 1
            bond_order = ob_bond.GetBondOrder()
            if bond_order == 1:
                rd_mol.AddBond(i, j, Chem.BondType.SINGLE)
            elif bond_order == 2:
                rd_mol.AddBond(i, j, Chem.BondType.DOUBLE)
            elif bond_order == 3:
                rd_mol.AddBond(i, j, Chem.BondType.TRIPLE)
            else:
                raise Exception('unknown bond order {}'.format(bond_order))
            if ob_bond.IsAromatic():
                bond = rd_mol.GetBondBetweenAtoms(i, j)
                bond.SetIsAromatic(True)
        rd_mol = Chem.RemoveHs(rd_mol, sanitize=False)
        return rd_mol

    def build_mol(self, xyz, atomic_nums):
        try:
            mol, atoms = self.make_obmol(xyz, atomic_nums)
            self.connect_the_dots(mol, atoms)
            rd_mol = self.convert_ob_mol_to_rd_mol(mol)
            return rd_mol
        except Exception as e:
            print(f"Error building molecule: {e}")
            return None

    def compute_iou(self, original_mol, generated_mol):
        mcs_result = rdFMCS.FindMCS([original_mol, generated_mol],timeout=2)
        mcs_atom_count = mcs_result.numAtoms
        original_heavy = original_mol.GetNumHeavyAtoms()
        generated_heavy = generated_mol.GetNumHeavyAtoms()
        union = original_heavy + generated_heavy - mcs_atom_count
        return mcs_atom_count / union if union != 0 else 0

    def reachable(self, a, b):
        if a.GetExplicitDegree() == 1 or b.GetExplicitDegree() == 1:
            return False
        seenbonds = set([a.GetBond(b).GetIdx()])
        return self.reachable_r(a, b, seenbonds)

    def reachable_r(self, a, b, seenbonds):
        for nbr in ob.OBAtomAtomIter(a):
            bond = a.GetBond(nbr).GetIdx()
            if bond not in seenbonds:
                seenbonds.add(bond)
                if nbr == b:
                    return True
                elif self.reachable_r(nbr, b, seenbonds):
                    return True
        return False

    def forms_small_angle(self, a, b, cutoff=60):
        for nbr in ob.OBAtomAtomIter(a):
            if nbr != b:
                degrees = b.GetAngle(a, nbr)
                if degrees < cutoff:
                    return True
        return False

    def count_nbrs_of_elem(self, atom, atomic_num):
        count = 0
        for nbr in ob.OBAtomAtomIter(atom):
            if nbr.GetAtomicNum() == atomic_num:
                count += 1
        return count