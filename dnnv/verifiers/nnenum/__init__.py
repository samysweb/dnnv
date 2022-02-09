import numpy as np
import os
import tempfile
import pickle

from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Type, Union

from dnnv.verifiers.common.base import Parameter, Verifier
from dnnv.verifiers.common.reductions import MixedConstraintIOPolytopeReduction, HalfspacePolytope
from dnnv.verifiers.common.results import SAT, UNSAT, UNKNOWN, PropertyCheckResult

from .errors import NnenumError, NnenumTranslatorError


class Nnenum(Verifier):
    reduction = partial(MixedConstraintIOPolytopeReduction, HalfspacePolytope, HalfspacePolytope)
    translator_error = NnenumTranslatorError
    verifier_error = NnenumError
    parameters = {
        "num_processes": Parameter(int, help="Maximum number of processes to use."),
        "iterate_violations": Parameter(bool, default=False, help="Iterate over violations.")
    }

    @contextmanager
    def contextmanager(self):
        orig_OPENBLAS_NUM_THREADS = os.getenv("OPENBLAS_NUM_THREADS")
        orig_OMP_NUM_THREADS = os.getenv("OMP_NUM_THREADS")
        try:
            os.environ["OPENBLAS_NUM_THREADS"] = "1"
            os.environ["OMP_NUM_THREADS"] = "1"
            yield
        finally:
            if orig_OPENBLAS_NUM_THREADS is not None:
                os.environ["OPENBLAS_NUM_THREADS"] = orig_OPENBLAS_NUM_THREADS
            else:
                del os.environ["OPENBLAS_NUM_THREADS"]
            if orig_OMP_NUM_THREADS is not None:
                os.environ["OMP_NUM_THREADS"] = orig_OMP_NUM_THREADS
            else:
                del os.environ["OMP_NUM_THREADS"]

    def build_inputs(self, prop):
        if prop.input_constraint.num_variables > 1:
            raise self.translator_error(
                "Unsupported network: More than 1 input variable"
            )

        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".onnx", delete=False
        ) as onnx_model_file:
            prop.op_graph.simplify().export_onnx(onnx_model_file.name)

        lb, ub = prop.input_constraint.as_bounds()
        A_in, b_in = prop.input_constraint.as_matrix_inequality()
        A_out, b_out = prop.output_constraint.as_matrix_inequality(include_bounds=True)

        with tempfile.NamedTemporaryFile(
            mode="wb+", suffix=".npy", delete=False
        ) as constraint_file:
            pickle.dump(((lb, ub), (A_in, b_in), (A_out, b_out)), constraint_file)

        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".npy", delete=False
        ) as output_file:
            self._tmp_output_file = output_file
        args = (
            "nnenum_counterex",
            onnx_model_file.name,
            constraint_file.name,
            "-o",
            self._tmp_output_file.name,
        )
        if (
            "num_processes" in self.parameters
            and self.parameters["num_processes"] is not None
        ):
            value = self.parameters["num_processes"]
            args += (f"--num_processes={value}",)
        if (
            "iterate_violations" in self.parameters
            and self.parameters["iterate_violations"] is not None
        ):
            args += (f"--iterate_violations",)
        return args

    def parse_results(self, prop, results):
        result_str, cex = np.load(self._tmp_output_file.name, allow_pickle=True)
        if result_str == "safe":
            return UNSAT, None
        elif result_str.startswith("unsafe"):
            return SAT, cex
        elif result_str == "error":
            raise self.verifier_error("result:error")
        raise self.translator_error(f"Unknown verification result: {result_str}")

    def run(self) -> Tuple[PropertyCheckResult, Optional[Any]]:
        if not self.parameters["iterate_violations"]:
            return super().run()
        # In this case, we need to iterate over all violations
        if self.property.is_concrete:
            if self.property.value == True:
                self.logger.warning("Property is trivially UNSAT.")
                return UNSAT, None
            else:
                self.logger.warning("Property is trivially SAT.")
                return SAT, None
        orig_tempdir = tempfile.tempdir
        cex = [None,[]]
        try:
            #with tempfile.TemporaryDirectory() as tempdir:
            tempfile.tempdir = "/tmp/cache" #tempdir
            result = UNSAT
            self.logger.debug("Beginning Property Reduction...")
            for subproperty in self.reduce_property():
                self.logger.debug("Checking next subproperty...")
                subproperty_result, cur_cex = self.check(subproperty)
                result |= subproperty_result
                if result == SAT and cur_cex is not None:
                        # Sometimes we get invalid counter-examples due to numerical issues
                        try:
                            self.validate_counter_example(subproperty, cur_cex[0])
                        except NnenumError as e:
                            self.logger.warn("SAT result without counter example.")
                            self.logger.warn(e)
                            cur_cex = (None,cur_cex[1])
                        if cur_cex[0] is not None:
                            # Concrete counter-example
                            self.logger.debug("SAT! Validated counter example.")
                            cex[0] = cur_cex[0]
                        # Adding any counter-example star sets there might be
                        self.logger.info(f"Encountered {len(cur_cex[1])} counter-example star sets.")
                        if len(cur_cex[1]) == 0:
                            self.logger.warn("No counter-example star sets.")
                        for star in cur_cex[1]:
                            cex[1].append(star)
                self.logger.debug(f"Continuing with next subproperty.")
        finally:
            tempfile.tempdir = orig_tempdir
        return result, cex