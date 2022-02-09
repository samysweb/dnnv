from __future__ import annotations

import copy
import logging
import numpy as np

from typing import Dict, Iterator, List, Tuple, Type, Union

from .base import Constraint, HalfspacePolytope, HyperRectangle, Variable
from .errors import IOPolytopeReductionError
from .property import IOPolytopeProperty
from .reduction import IOPolytopeReduction
from ..base import Property, Reduction, ReductionError
from .....properties import *
from .....properties.transformers.base import GenericExpressionTransformer
from .....nn.transformers import OperationTransformer
from .....nn.operations import Input, Gemm, Relu
from .....nn.graph import OperationGraph


class MixedConstraintIOPolytopeReduction(IOPolytopeReduction):
    def __init__(
        self,
        input_constraint_type: Type[Constraint] = HyperRectangle,
        output_constraint_type: Type[Constraint] = HalfspacePolytope,
        reduction_error: Type[ReductionError] = IOPolytopeReductionError,
    ):
        super().__init__(
            input_constraint_type=input_constraint_type,
            output_constraint_type=output_constraint_type,
            reduction_error=reduction_error)
        self.unmix_transformer = None
    
    def reduce_property(self, expression: Expression) -> Iterator[Property]:
        if not isinstance(expression, Exists):
            raise NotImplementedError()  # TODO
        dnf_expression = expression.canonical()
        assert isinstance(dnf_expression, Or)
        netTransformer = NetworkTransformer()
        dnf_expression = netTransformer.visit(dnf_expression)
        for conjunction in dnf_expression:
            yield from self._reduce(conjunction)
    
    def unmix_conjunction(self, conjunction: Expression) -> Expression:
        self.unmix_transformer = UnmixConstraintsTransformer(
            self._network_input_shapes,self._network_output_shapes)
        return self.unmix_transformer.visit(conjunction)

class NetworkTransformer(GenericExpressionTransformer):
    """
    This transformer propagates the inputs through the network for all networks mentioned in the expression.
    Note, that this transformer must only be used once.
    """

    def __init__(self):
        super().__init__()
        self._networks = {}

    @property
    def networks(self):
        return self._networks

    def visit_Network(self, expression: Network):
        graph = expression.value
        if graph not in self._networks:
            self._networks[graph] = self.propagate_inputs_through_network(graph)
        expression.concretize(self._networks[graph])
        return expression
    
    def propagate_inputs_through_network(self, graph: OperationGraph) -> OperationGraph:
        if len(graph.input_shape) != 1 or len(graph.output_shape) != 1:
            raise NotImplementedError("Only supporting single in/output networks")
        if len(graph.input_shape[0]) != 2 or graph.input_shape[0][0] != 1\
            or len(graph.output_shape[0]) != 2 or graph.output_shape[0][0] != 1:
            raise NotImplementedError(f"Only networks with (1,n) inputs and (1,m) outputs are supported. This network has {graph.input_shape[0]} and {graph.output_shape[0]}!")
        transformer = PropagateInputsOperationsTransformer()
        return transformer.visit(graph)

class PropagateInputsOperationsTransformer(OperationTransformer):
    def __init__(self):
        super().__init__()
        self.input_size = None
    
    def visit_Input(self, operation: Input) -> Input:
        if len(operation.shape) != 2 or operation.shape[0] != 1:
            raise NotImplementedError(f"Only networks with (1,n) inputs and (1,m) outputs are supported. This network has a {operation.shape[0]} input!")
        self.input_size = operation.shape[1]
        return operation

    def visit_Gemm(self, operation: Gemm) -> Gemm:
        new_a = self.visit(operation.a)
        assert self.input_size is not None, "Input size not set!"
        if isinstance(new_a, Input):
            input_identity = np.identity(self.input_size, dtype=operation.b.dtype)
            new_b = np.vstack((operation.b, input_identity, -input_identity))
        else:
            zero_b = np.zeros((2*self.input_size,operation.b.shape[1]), dtype=operation.b.dtype)
            input_identity = np.vstack((
                np.zeros((operation.b.shape[0],2*self.input_size), dtype=operation.b.dtype),
                np.identity(2*self.input_size, dtype=operation.b.dtype)
            ))
            new_b = np.hstack((
                np.vstack((operation.b, zero_b)),
                input_identity
            ))
        new_c = np.hstack((operation.c, np.zeros((2*self.input_size), dtype=operation.b.dtype)))
        return Gemm(new_a, new_b, new_c)
    
    def visit_Relu(self, operation: Relu) -> Relu:
        new_node = self.visit(operation.x)
        return Relu(new_node)

    def generic_visit(self, operation: Operation) -> Operation:
        if isinstance(operation, OperationGraph):
            return super().generic_visit(operation)
        raise NotImplementedError(f"Operation {operation} not supported!")



class UnmixConstraintsTransformer(GenericExpressionTransformer):
    """
    This transformer resolves mixed constraints.
    Note, that each transformer instance must be used only once.
    """
    def __init__(self,
    network_input_shapes:Dict[Expression, Tuple[int, ...]],
    network_output_shapes:Dict[Expression, Tuple[int, ...]]):
        super().__init__()
        self._network_input_shapes = network_input_shapes
        self._network_output_shapes = network_output_shapes
        self.unmix_necessary = False
        self.networks = None

    def visit(self, expression: Expression) -> Expression:
        if self._top_level:
            # TODO
            print("Unmix Transformer")
        expression = super().visit(expression)
        return expression
    
    def visit_And(self, expression: And) -> Expression:
        expressions = []
        for expr in expression.expressions:
            self.networks = None
            assert len(expr.networks) <= 1, "Currently only supporting problems with one network"
            if len(expr.networks) == 1:
                # In this case we need to resolve possible input variables (and reformat network outputs)
                # Store networks to figure out shapes later
                self.networks = [n for n in expr.networks]
                expr = self.visit(expr)
            if isinstance(expr, And):
                expressions.extend(expr.expressions)
            else:
                expressions.append(expr)
        return And(*expressions)

    def visit_Call(self, expression: Call) -> Expression:
        if isinstance(expression.function, Network):
            input_details = expression.function.value.input_details
            assert(len(expression.function.value.output_shape[0])==2 and expression.function.value.output_shape[0][0]==1), "Currently only supporting mixed constraints for 2D output networks with dimension 1 being 1"
            assert(len(expression.function.value.input_shape[0])==2 and expression.function.value.input_shape[0][0]==1), "Currently only supporting mixed constraints for 2D input networks with dimension 1 being 1"
            regularEnd1 = expression.function.value.output_shape[0][0]
            regularEnd2 = expression.function.value.output_shape[0][1]
            inputSize = expression.function.value.input_shape[0][1]
            
            expression.function._value.output_shape = (
                (regularEnd1, regularEnd2+inputSize),
            )

            return expression[:regularEnd1,:regularEnd2]
        else:
            raise self.reduction_error(
                "Unsupported property:"
                f" Function {expression.function} is not currently supported"
            )
        
    def visit_Network(self, expression: Network) -> Expression:
        expression.value.output_shape = (
            (expression.value.output_shape[0][0],
            expression.value.output_shape[0][1]+self._network_input_shapes[expression][1]),
        )
        return expression
    
    def visit_Symbol(self, expression: Symbol):
        if expression not in self._network_input_shapes:
            print("HELP! I'm lost, what symbol is this?")
        print("Oh look at this mixed variable! We need to resolve it!")
        print(expression)
        #print("Shape: ",self._network_input_shapes[expression])
        print("Output Shape: ",self.networks[0].value.output_shape)
        self.unmix_necessary = True
        assert len(self.networks[0].value.output_shape[0])==2 and self.networks[0].value.output_shape[0][0] == 1, "Currently only supporting mixed constraints for 2D output networks with dimension 1 being 1"
        callexpr = (Call(
            function=self.networks[0],
            args=[expression],
            kwargs={},
        ))[:self.networks[0].value.output_shape[0][0],self.networks[0].value.output_shape[0][1]:]
        #self._network_output_shapes[self.networks[0]] = (
        #    (self.networks[0].value.output_shape[0][0],
        #    self.networks[0].value.output_shape[0][1]+self._network_input_shapes[expression][1])
        #)
        print("New shape:")
        print(self._network_output_shapes[self.networks[0]])
        return callexpr

        
