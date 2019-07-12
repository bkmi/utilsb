import numpy as np
import mdtraj
import tempfile
import subprocess
import os

from typing import List, Union, Optional, Tuple


VMD_EXE = "/Applications/VMD 1.9.3.app/Contents/vmd/vmd_MACOSXX86"


def symbol_to_sequential(include_hydrogens: bool = False):
    sequential_map = {
        'C': 0,
        'N': 1,
        'O': 2,
        'F': 3,
        '-': 4
    }
    if include_hydrogens:
        sequential_map['H'] = 5
    return sequential_map


def symbol_to_one_hot(include_hydrogens: bool = False, include_blank: bool = True):
    one_hot = {
        'C': [1, 0, 0, 0, ],
        'N': [0, 1, 0, 0, ],
        'O': [0, 0, 1, 0, ],
        'F': [0, 0, 0, 1, ],
    }
    if include_blank:
        for k in one_hot.keys():
            one_hot[k].append(0)
        # one_hot['-'] = [0, 0, 0, 0, 1]
        one_hot['-'] = [0] * (len(one_hot['F']) - 1) + [1]

    if include_hydrogens:
        for k in one_hot.keys():
            one_hot[k].append(0)
        # one_hot['H'] = [0, 0, 0, 0, 0, 1]
        one_hot['H'] = [0] * (len(one_hot['F']) - 1) + [1]
    return one_hot


def atomic_number_to_symbol():
    return {0: '-', 1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}


# You can load multiple molecules using https://www.ks.uiuc.edu/Research/vmd/mailing_list/vmd-l/13045.html
# cat ~bin/load-pdbs.vmd
# foreach i $argv {
#   mol new $i
# }
# vmd -e ~/bin/load-pdbs.vmd -args one.pdb two.pdb three.pdb otherdirctory/*.pdb
# make each molecule a separate number (kinda done but does not work with qm9.buf style)
# TODO make Cs atoms invisible
class Scene(object):
    """Designed to work with qm9.buf types format."""
    def __init__(self, filename: Optional[str] = None, include_hydrogens: bool = True):
        if filename is None:
            self.file = tempfile.NamedTemporaryFile()
            self.filename = self.file.name
        else:
            self.file = None
            self.filename = filename
        self.include_hydrogens = include_hydrogens

        self.traj_coords = []
        self._max_natoms = None

        self.top = mdtraj.Topology()
        self.chain = self.top.add_chain()
        self.process = None

    def __call__(self, coords: Union[np.ndarray, List[np.ndarray]], types: Union[np.ndarray, List[np.ndarray]],
                 scale: float = 0.1, *args, **kwargs):
        self.coords = self._process_input(coords)
        self.types = self._process_input(types)

        self.place_molecules(self.coords, self.types)
        self.traj = self._make_trajectory(scale=scale)
        self.save_trajectory(self.filename)
        self.show_vmd(self.filename)

    @staticmethod
    def sequential_to_atomic_number(include_hydrogens: bool = False):
        an_mapping = {
            0: 6,
            1: 7,
            2: 8,
            3: 9,
            4: 0
        }
        if include_hydrogens:
            an_mapping[5] = 1
        return an_mapping

    @staticmethod
    def _process_input(array: Union[np.ndarray, List[np.ndarray]]):
        if type(array) == np.ndarray:
            return [i.squeeze() for i in np.split(array, array.shape[0])]
        elif type(array) == list:
            return array
        else:
            raise TypeError('Input must be list with each element shape == (natom, nfeatures) or ndarray.')

    @property
    def max_natoms(self):
        if self._max_natoms is None:
            natoms = 0
            for c, t in zip(self.coords, self.types):
                if c.shape[0] > natoms:
                    natoms = c.shape[0]
                assert c.shape[0] <= natoms
                assert t.shape[0] <= natoms
            self._max_natoms = natoms
        return self._max_natoms

    def _place_Cs(self, res):
        self.top.add_atom('Cs', mdtraj.element.cesium, res)
        self.traj_coords.append(np.zeros_like(self.traj_coords[-1]))

    def place_molecules(self, coords: List[np.ndarray], types: List[np.ndarray]):
        for i, (c, t) in enumerate(zip(coords, types)):
            assert c.ndim == 2
            assert c.shape[1] == 3
            assert t.ndim == 2
            # assert c.shape[0] == t.shape[0]

            res = self.top.add_residue("mol_{}".format(i), self.chain)
            t = np.argmax(t.copy(), axis=1)
            t_atomic = [self.sequential_to_atomic_number(self.include_hydrogens)[tt] for tt in t]

            # Place every atom in the scene.
            # When there is a '-' type, place a special atom at the center.
            # When there is an atom missing, place a special atom at the center
            for j in range(self.max_natoms):
                try:
                    if t_atomic[j] != 0:
                        symbol = atomic_number_to_symbol()[t_atomic[j]]
                        self.top.add_atom(symbol, mdtraj.element.get_by_symbol(symbol), res)
                        self.traj_coords.append(c[j])
                    else:
                        self._place_Cs(res)
                except IndexError:
                    self._place_Cs(res)

    def _make_trajectory(self, scale: float = 0.1):
        return mdtraj.Trajectory(scale * np.stack(self.traj_coords), self.top)
        # return mdtraj.Trajectory(scale * np.stack(self.traj_coords).reshape(-1, self.max_natoms, 3), self.top)

    def save_trajectory(self, filename: str, force_overwrite: bool = True):
        self.traj.save_pdb(filename, force_overwrite=force_overwrite)

    def show_vmd(self, filename: Optional[str] = None):
        if filename is None:
            message = ['vmd', self.file.name]
        else:
            message = ['vmd', str(filename)]

        if self.process is None:
            self.process = subprocess.Popen(message, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE)
        else:
            self.process.kill()
            self.process = subprocess.Popen(message, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.process.wait()


class Molecules(object):
    """Designed to work with regular qm9 format."""
    def __init__(self, coords: Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray]],
                 types: Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray]],
                 include_hydrogens: bool = True,
                 scale: float = 0.1):
        self.process = None

        # Setup files and mdtraj for each molecule
        assert len(coords) == len(types)
        self.files, self.tops, self.trajs = [], [], []
        for _ in range(len(coords)):
            self.files.append(tempfile.NamedTemporaryFile())
            self.tops.append(mdtraj.Topology())
        self.include_hydrogens = include_hydrogens

        # Populate Trajectories
        self.coords = [c.squeeze() for c in coords]
        self.types = [t.squeeze() for t in types]

        for c, typ, f, to in zip(self.coords, self.types, self.files, self.tops):
            self.add_mol_to_topology(c, typ, to)
            self.trajs.append(self._make_trajectory(coords=c, topology=to, scale=scale))
            self.save_trajectory(trajectory=self.trajs[-1], filename=f.name)
        self.show_vmd()

    @staticmethod
    def sequential_to_atomic_number():
        an_mapping = {
            0: 6,
            1: 7,
            2: 8,
            3: 9,
            4: 1
        }
        return an_mapping

    def add_mol_to_topology(self, coords: np.ndarray, types: np.ndarray, topology: mdtraj.Topology):
        assert coords.shape[0] == types.shape[0]
        assert coords.ndim == 2
        chain = topology.add_chain()

        # Convert types to symbols
        if types.ndim == 2:
            seqs = np.argmax(types.copy(), axis=1)
            atms = [self.sequential_to_atomic_number()[t] for t in seqs]
            syms = [atomic_number_to_symbol()[t] for t in atms]
        elif types.ndim == 1:
            syms = [atomic_number_to_symbol()[t] for t in types]
        else:
            raise ValueError("Types must either be one hot vectors with ndim==2 XOR numbers with ndim==1.")

        for i, s in enumerate(syms):
            res = topology.add_residue("mol_{}".format(i), chain)
            topology.add_atom(s, mdtraj.element.get_by_symbol(s), res)

    @staticmethod
    def _make_trajectory(coords: np.ndarray, topology: mdtraj.Topology, scale: float = 0.1):
        assert coords.ndim == 2
        assert coords.shape[1] == 3
        return mdtraj.Trajectory(scale * coords, topology)

    @staticmethod
    def save_trajectory(trajectory: mdtraj.Trajectory, filename: str, force_overwrite: bool = True):
        trajectory.save_pdb(filename, force_overwrite=force_overwrite)

    def show_vmd(self):
        # load several pdbs with:
        # vmd -e ~/bin/load-pdbs.vmd -args one.pdb two.pdb three.pdb otherdirctory/*.pdb
        script = os.path.dirname(os.path.realpath(__file__)) + '/' + 'multipdb.sh'
        message = [VMD_EXE, '-e', script, '-args'] + [f.name for f in self.files]

        if self.process is None:
            self.process = subprocess.Popen(message, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE)
        else:
            self.process.kill()
            self.process = subprocess.Popen(message, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.process.wait()


class InterpolateTrajectory(object):
    """Designed to work with regular qm9 format."""
    def __init__(self, coords: Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray]],
                 types: Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray]],
                 include_hydrogens: bool = True,
                 scale: float = 0.1):
        self.process = None

        # Setup files and mdtraj for each molecule
        assert len(coords) == len(types)
        self.files, self.tops, self.trajs = [], [], []
        for _ in range(len(coords)):
            self.files.append(tempfile.NamedTemporaryFile())
            self.tops.append(mdtraj.Topology())
        self.include_hydrogens = include_hydrogens

        # Populate Trajectories
        self.coords = [c.squeeze() for c in coords]
        self.types = [t.squeeze() for t in types]

        for c, typ, f, to in zip(self.coords, self.types, self.files, self.tops):
            self.add_mol_to_topology(c, typ, to)
            self.trajs.append(self._make_trajectory(coords=c, topology=to, scale=scale))
            self.save_trajectory(trajectory=self.trajs[-1], filename=f.name)
        self.show_vmd()

    @staticmethod
    def sequential_to_atomic_number():
        an_mapping = {
            0: 6,
            1: 7,
            2: 8,
            3: 9,
            4: 1
        }
        return an_mapping

    def add_mol_to_topology(self, coords: np.ndarray, types: np.ndarray, topology: mdtraj.Topology):
        assert coords.shape[0] == types.shape[0]
        assert coords.ndim == 2
        chain = topology.add_chain()

        # Convert types to symbols
        if types.ndim == 2:
            seqs = np.argmax(types.copy(), axis=1)
            atms = [self.sequential_to_atomic_number()[t] for t in seqs]
            syms = [atomic_number_to_symbol()[t] for t in atms]
        elif types.ndim == 1:
            syms = [atomic_number_to_symbol()[t] for t in types]
        else:
            raise ValueError("Types must either be one hot vectors with ndim==2 XOR numbers with ndim==1.")

        for i, s in enumerate(syms):
            res = topology.add_residue("mol_{}".format(i), chain)
            topology.add_atom(s, mdtraj.element.get_by_symbol(s), res)

    @staticmethod
    def _make_trajectory(coords: np.ndarray, topology: mdtraj.Topology, scale: float = 0.1):
        assert coords.ndim == 2
        assert coords.shape[1] == 3
        return mdtraj.Trajectory(scale * coords, topology)

    @staticmethod
    def save_trajectory(trajectory: mdtraj.Trajectory, filename: str, force_overwrite: bool = True):
        trajectory.save_pdb(filename, force_overwrite=force_overwrite)

    def show_vmd(self):
        # load several pdbs with:
        # vmd -e ~/bin/load-pdbs.vmd -args one.pdb two.pdb three.pdb otherdirctory/*.pdb
        script = os.path.dirname(os.path.realpath(__file__)) + '/' + 'multipdb.sh'
        message = ['vmd', '-e', script, '-args'] + [f.name for f in self.files]

        if self.process is None:
            self.process = subprocess.Popen(message, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE)
        else:
            self.process.kill()
            self.process = subprocess.Popen(message, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.process.wait()


def main_scene():
    import pdbpredict_tf.data.qm9data as qm9

    include_hydrogens = True

    coords, types = qm9.get_coords_and_types(include_hydrogens=include_hydrogens)
    coords = coords[:100]
    types = types[:100]

    # coords = [i.squeeze() for i in np.split(coords, coords.shape[0])]
    # types = [i.squeeze() for i in np.split(types, types.shape[0])]

    s = Scene(include_hydrogens=include_hydrogens)
    s(coords, types)


def main_molecules():
    import pickle

    qm9pickle = "/home/ben/science/data/limited_qm9.pickle"
    with open(qm9pickle, 'rb') as f:
        out = pickle.load(f)

    types = out['train_oht'][:10]
    coords = out['train_coords'][:10]

    Molecules(coords, types)


if __name__ == '__main__':
    # main_scene()
    main_molecules()
