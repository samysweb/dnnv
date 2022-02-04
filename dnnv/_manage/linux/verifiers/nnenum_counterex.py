from __future__ import annotations

import subprocess as sp
import sys

from ..environment import (
    Environment,
    Dependency,
    ProgramDependency,
    Installer,
)
from ...errors import InstallError, UninstallError

runner_template = """#!{python_venv}/bin/python
import argparse
import numpy as np
import pickle

from pathlib import Path

from nnenum.enumerate import enumerate_network
from nnenum.lp_star import LpStar
from nnenum.onnx_network import load_onnx_network
from nnenum.settings import Settings
from nnenum.specification import Specification


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("constraints")

    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("-p", "--num_processes", "--procs", type=int)
    parser.add_argument("--iterate_violations", action='store_true')

    return parser.parse_args()


def main(args):
    Settings.UNDERFLOW_BEHAVIOR = "warn"
    Settings.PRINT_PROGRESS = False
    Settings.PRINT_OUTPUT = False
    if args.iterate_violations:
        Settings.RESULT_SAVE_COUNTER_STARS = True
    Settings.FIND_CONCRETE_COUNTEREXAMPLES = True
    if args.num_processes is not None:
        Settings.NUM_PROCESSES = args.num_processes

    (lb, ub), (A_input, b_input), (A_output, b_output) = (None,None), (None,None), (None,None)
    with open(args.constraints, "rb") as f:
        (lb, ub), (A_input, b_input), (A_output, b_output) = pickle.load(f)
    network = load_onnx_network(args.model)
    ninputs = A_input.shape[1]

    init_box = np.array(
        list(zip(lb.flatten(), ub.flatten())),
        dtype=np.float32,
    )
    init_star = LpStar(
        np.eye(ninputs, dtype=np.float32), np.zeros(ninputs, dtype=np.float32), init_box
    )
    for a, b in zip(A_input, b_input):
        a_ = a.reshape(network.get_input_shape()).flatten("F")
        init_star.lpi.add_dense_row(a_, b)

    spec = Specification(A_output, b_output)

    result = enumerate_network(init_star, network, spec)

    print(result.result_str)
    if args.output is not None:
        cex = None
        counterex_stars = []
        if result.cinput is not None:
            cex = (
                np.array(list(result.cinput))
                .astype(np.float32)
                .reshape(network.get_input_shape())
            )
            print(cex)
            for star in result.stars:
                # Extract Ax <= b
                A = star.lpi.get_constraints_csr()
                b = star.lpi.get_rhs()
                # Extract M*x + c
                M = star.a_mat
                c = star.bias
                # Compute bounds
                dims = star.lpi.get_num_cols()
                should_skip = np.zeros((dims, 2), dtype=bool)
                bounds = star.update_input_box_bounds_old(None, should_skip)
                counterex_stars.append((
                    A, b,
                    M, c,
                    bounds
                ))
        np.save(args.output, (result.result_str, (cex, counterex_stars)))

    return


if __name__ == "__main__":
    main(parse_args())
"""


class NnenumCounterexInstaller(Installer):
    def run(self, env: Environment, dependency: Dependency):
        commit_hash = "fd07f2b6c55ca46387954559f40992ae0c9b06b7"
        name = "nnenum_counterex"

        cache_dir = env.cache_dir / f"{name}-{commit_hash}"
        cache_dir.mkdir(exist_ok=True, parents=True)

        installation_path = env.env_dir / "bin"
        installation_path.mkdir(exist_ok=True, parents=True)

        verifier_venv_path = env.env_dir / "verifier_virtualenvs" / name
        verifier_venv_path.parent.mkdir(exist_ok=True, parents=True)

        python_major_version, python_minor_version = sys.version_info[:2]

        envvars = env.vars()
        commands = [
            "set -ex",
            f"cd {verifier_venv_path.parent}",
            f"rm -rf {name}",
            f"python -m venv {name}",
            f". {name}/bin/activate",
            "pip install --upgrade pip",
            f"cd {cache_dir}",
            f"rm -rf {name}",
            f"git clone https://github.com/samysweb/nnenum.git {name}",
            f"cd {name}",
            f"git checkout {commit_hash}",
            # "pip install -r requirements.txt",
            'pip install "numpy>=1.19,<1.22" "onnx>=1.8,<1.11" "onnxruntime>=1.7,<1.11" "scipy>=1.4.1<1.8" "threadpoolctl==2.1.0" "skl2onnx==1.7.0" "swiglpk" "termcolor"',
            f"ln -s {cache_dir}/{name}/src/nnenum {verifier_venv_path}/lib/python{python_major_version}.{python_minor_version}/site-packages/nnenum",
        ]
        install_script = "; ".join(commands)
        proc = sp.run(install_script, shell=True, env=envvars)
        if proc.returncode != 0:
            raise InstallError(f"Installation of {name} failed")

        with open(installation_path / name, "w+") as f:
            f.write(runner_template.format(python_venv=verifier_venv_path))
        (installation_path / name).chmod(0o700)


def install(env: Environment):
    env.ensure_dependencies(
        ProgramDependency(
            "nnenum_counterex",
            installer=NnenumCounterexInstaller(),
            dependencies=(ProgramDependency("git"),),
        )
    )


def uninstall(env: Environment):
    name = "nnenum_counterex"
    exe_path = env.env_dir / "bin" / name
    verifier_venv_path = env.env_dir / "verifier_virtualenvs" / name
    commands = [
        f"rm -f {exe_path}",
        f"rm -rf {verifier_venv_path}",
    ]
    install_script = "; ".join(commands)
    proc = sp.run(install_script, shell=True, env=env.vars())
    if proc.returncode != 0:
        raise UninstallError(f"Uninstallation of {name} failed")


__all__ = ["install", "uninstall"]
