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
        with expression.ctx:
            dnf_expression = netTransformer.visit(dnf_expression)
            dnf_expression = self.unmix_conjunction(
                dnf_expression,
                netTransformer.networks,
                netTransformer.inputs)
        for conjunction in dnf_expression:
            yield from self._reduce(conjunction)
    
    def unmix_conjunction(self,
        conjunction: Expression,
        networks: Dict[Expression, Tuple[Tuple[Tuple[int, ...],...], Tuple[Tuple[int, ...],...]]],
        inputs: Dict[Expression, Network]) -> Expression:
        self.unmix_transformer = UnmixConstraintsTransformer(networks, inputs)
        return self.unmix_transformer.visit(conjunction)

class NetworkTransformer(GenericExpressionTransformer):
    """
    This transformer propagates the inputs through the network for all networks mentioned in the expression.
    Note, that this transformer must only be used once.
    """

    def __init__(self):
        super().__init__()
        self._networks = {}
        self._inputs = {}

    @property
    def networks(self):
        return self._networks
    
    @property
    def inputs(self):
        return self._inputs
    
    def visit_Call(self, expression: Call):
        if isinstance(expression.function, Network):
            new_network = self.visit(expression.function)
            for x in expression.args:
                self._inputs[x]=new_network
            new_call = Call(
                function=new_network,
                args=expression.args,
                kwargs=expression.kwargs
            )
            expression.ctx.shapes[new_call] = self.networks[new_network][1][1]
            return new_call
        else:
            return expression


    def visit_Network(self, expression: Network):
        graph = expression.value
        new_network = self.propagate_inputs_through_network(graph)
        new_network_symbol = Network(expression.identifier)
        new_network_symbol.concretize(new_network)
        self._networks[new_network_symbol] = (
            # Old Shapes:
            (graph.input_shape[0], graph.output_shape[0]),
            # New Shapes:
            (new_network.input_shape[0], new_network.output_shape[0]),
        )
        return new_network_symbol
    
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
        result = Gemm(new_a, new_b, new_c,transpose_b=True)
        return result
    
    def visit_Relu(self, operation: Relu) -> Relu:
        new_node = self.visit(operation.x)
        return Relu(new_node)

    def visit_OperationGraph(self, operation: OperationGraph) -> OperationGraph:
        outputs = []
        for o in operation.output_operations:
            outputs.append(self.visit(o))
        return OperationGraph(outputs)

    def generic_visit(self, operation: Operation) -> Operation:
        raise NotImplementedError(f"Operation {operation} not supported!")



class UnmixConstraintsTransformer(GenericExpressionTransformer):
    """
    This transformer resolves mixed constraints.
    Note, that each transformer instance must be used only once.
    """
    def __init__(self,
        networks: Dict[Expression, Tuple[Tuple[Tuple[int, ...],...], Tuple[Tuple[int, ...],...]]],
        inputs: Dict[Expression, Network]):
        super().__init__()
        # Mapping of networks to their input shapes
        self.networks = networks
        # Mapping of inputs to their network
        self.inputs = inputs
        self.need_normalization = False
        self.need_normalization_cache = {}
        self.subscript_transformer = ResolveSubscripts()
    
    def visit_And(self, expression: And) -> Expression:
        expressions = []
        for expr in expression.expressions:
            assert len(expr.networks) <= 1, "Currently only supporting problems with one network"
            if len(expr.networks) == 1:
                old_expr = expr
                expr = self.visit(expr)
                if self.need_normalization:
                    self.logger.debug(f"Normalizing expression {expr}")
                    expr = self.subscript_transformer.visit(expr)
                    #expr = expr.canonical()
                    #if isinstance(expr, Or):
                    #    assert len(expr.expressions)==1, "Or expressions should have only one expression at this point"
                    #    expr = expr.expressions[0].expressions[0]
                    self.logger.debug(f"After normalization: {expr}")
                    self.need_normalization = False
                self.visited[old_expr] = expr
            if isinstance(expr, And):
                expressions.extend(expr.expressions)
            else:
                expressions.append(expr)
        result = And(*expressions)
        return result

    def visit_Call(self, expression: Call) -> Expression:
        if isinstance(expression.function, Network):
            (old_input_shape, old_output_shape) = self.networks[expression.function][0]
            assert (len(old_output_shape)==2)

            return expression[:old_output_shape[0],:old_output_shape[1]]
        else:
            return expression
    
    def visit_Symbol(self, expression: Symbol):
        if expression not in self.inputs:
            logging.debug(f"Ignoring symbol {expression} which does not seem to be a network input...")
        # If it is a network input, use the newly added propagated values...
        relevant_network = self.inputs[expression]
        first_slice_end = (self.networks[relevant_network][0][1][1]+self.networks[relevant_network][1][0][1])
        callexpr = (Call(
            function=relevant_network,
            args=[expression],
            kwargs={},
        ))
        expression.ctx.shapes[callexpr] = self.networks[relevant_network][1][1]
        inputexpr=Add(
            callexpr[:self.networks[relevant_network][0][1][0],self.networks[relevant_network][0][1][1]:first_slice_end],
            -callexpr[:self.networks[relevant_network][0][1][0],first_slice_end:]
        )
        expression.ctx.shapes[inputexpr] = self.networks[relevant_network][0][0]
        expression.ctx.types[inputexpr] = expression.ctx.types[expression]
        self.need_normalization = True
        return inputexpr

class ResolveSubscripts(GenericExpressionTransformer):
    def visit_Subscript(self, expression: Subscript) -> Expression:
        if isinstance(expression.expr1, Negation):
            new_inner_expr = Subscript(expression.expr1.expr, expression.expr2)
            new_inner_expr = self.visit(new_inner_expr)
            new_expr1 = Negation(new_inner_expr)
            return new_expr1
        elif isinstance(expression.expr1, Add):
            expressions = []
            for expr in expression.expr1.expressions:
                new_expr = self.visit(Subscript(expr, expression.expr2))
                expressions.append(new_expr)
            result = Add(*expressions)
            return result
        return expression
        
