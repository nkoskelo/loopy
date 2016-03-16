"""Target for Intel ISPC."""

from __future__ import division, absolute_import

__copyright__ = "Copyright (C) 2015 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


import numpy as np  # noqa
from loopy.target.c import CTarget
from loopy.target.c.codegen.expression import LoopyCCodeMapper
from loopy.diagnostic import LoopyError

from pytools import memoize_method


# {{{ expression mapper

class LoopyISPCCodeMapper(LoopyCCodeMapper):
    def _get_index_ctype(self):
        if self.kernel.index_dtype.numpy_dtype == np.int32:
            return "int32"
        elif self.kernel.index_dtype.numpy_dtype == np.int64:
            return "int64"
        else:
            raise ValueError("unexpected index_type")

    def map_group_hw_index(self, expr, enclosing_prec, type_context):
        return "((uniform %s) taskIndex%d)" % (self._get_index_ctype(), expr.axis)

    def map_local_hw_index(self, expr, enclosing_prec, type_context):
        if expr.axis == 0:
            return "(varying %s) programIndex" % self._get_index_ctype()
        else:
            raise LoopyError("ISPC only supports one local axis")

# }}}


# {{{ type registry

def fill_registry_with_ispc_types(reg, respect_windows, include_bool=True):
    reg.get_or_register_dtype("bool", np.bool)

    reg.get_or_register_dtype(["int8", "signed char", "char"], np.int8)
    reg.get_or_register_dtype(["uint8", "unsigned char"], np.uint8)
    reg.get_or_register_dtype(["int16", "short", "signed short",
        "signed short int", "short signed int"], np.int16)
    reg.get_or_register_dtype(["uint16", "unsigned short",
        "unsigned short int", "short unsigned int"], np.uint16)
    reg.get_or_register_dtype(["int32", "int", "signed int"], np.int32)
    reg.get_or_register_dtype(["uint32", "unsigned", "unsigned int"], np.uint32)

    reg.get_or_register_dtype(["int64"], np.int64)
    reg.get_or_register_dtype(["uint64"], np.uint64)

    reg.get_or_register_dtype("float", np.float32)
    reg.get_or_register_dtype("double", np.float64)

# }}}


class ISPCTarget(CTarget):
    """A code generation target for Intel's `ISPC <https://ispc.github.io/>`_
    SPMD programming language, to target Intel's Knight's hardware and modern
    Intel CPUs with wide vector units.
    """

    def __init__(self, occa_mode=False):
        """
        :arg occa_mode: Whether to modify the generated call signature to
            be compatible with OCCA
        """
        self.occa_mode = occa_mode

        super(ISPCTarget, self).__init__()

    # {{{ types

    @memoize_method
    def get_dtype_registry(self):
        from loopy.target.c.compyte.dtypes import DTypeRegistry
        result = DTypeRegistry()
        fill_registry_with_ispc_types(result, respect_windows=False,
                include_bool=True)
        return result

    # }}}

    # {{{ top-level codegen

    def generate_code(self, kernel, codegen_state, impl_arg_info):
        from cgen import (FunctionBody, FunctionDeclaration, Value, Module,
                Block, Line, Statement as S)
        from cgen.ispc import ISPCExport, ISPCTask

        knl_body, implemented_domains = kernel.target.generate_body(
                kernel, codegen_state)

        inner_name = "lp_ispc_inner_"+kernel.name
        arg_decls = [iai.cgen_declarator for iai in impl_arg_info]
        arg_names = [iai.name for iai in impl_arg_info]

        # {{{ occa compatibility hackery

        if self.occa_mode:
            from cgen import ArrayOf, Const
            from cgen.ispc import ISPCUniform

            arg_decls = [
                    Const(ISPCUniform(ArrayOf(Value("int", "loopy_dims")))),
                    Const(ISPCUniform(Value("int", "o1"))),
                    Const(ISPCUniform(Value("int", "o2"))),
                    Const(ISPCUniform(Value("int", "o3"))),
                    ] + arg_decls
            arg_names = ["loopy_dims", "o1", "o2", "o3"] + arg_names

        # }}}

        knl_fbody = FunctionBody(
                ISPCTask(
                    FunctionDeclaration(
                        Value("void", inner_name),
                        arg_decls)),
                knl_body)

        # {{{ generate wrapper

        wrapper_body = Block()

        gsize, lsize = kernel.get_grid_sizes_as_exprs()
        if len(lsize) > 1:
            for i, ls_i in enumerate(lsize[1:]):
                if ls_i != 1:
                    raise LoopyError("local axis %d (0-based) "
                            "has length > 1, which is unsupported "
                            "by ISPC" % ls_i)

        from pymbolic.mapper.stringifier import PREC_COMPARISON, PREC_NONE
        ccm = self.get_expression_to_code_mapper(codegen_state)

        if lsize:
            wrapper_body.append(
                    S("assert(programCount == %s)"
                        % ccm(lsize[0], PREC_COMPARISON)))

        if gsize:
            launch_spec = "[%s]" % ", ".join(
                                ccm(gs_i, PREC_NONE)
                                for gs_i in gsize),
        else:
            launch_spec = ""

        wrapper_body.append(
                S("launch%s %s(%s)"
                    % (
                        launch_spec,
                        inner_name,
                        ", ".join(arg_names)
                        ))
                )

        wrapper_fbody = FunctionBody(
                ISPCExport(
                    FunctionDeclaration(
                        Value("void", kernel.name),
                        arg_decls)),
                wrapper_body)

        # }}}

        mod = Module([
            knl_fbody,
            Line(),
            wrapper_fbody,
            ])

        return str(mod), implemented_domains

    # }}}

    # {{{ code generation guts

    def get_expression_to_code_mapper(self, codegen_state):
        return LoopyISPCCodeMapper(codegen_state)

    def add_vector_access(self, access_str, index):
        return "(%s)[%d]" % (access_str, index)

    def emit_barrier(self, kind, comment):
        from loopy.codegen import GeneratedInstruction
        from cgen import Comment, Statement

        assert comment

        if kind == "local":
            return GeneratedInstruction(
                    ast=Comment("local barrier: %s" % comment),
                    implemented_domain=None)

        elif kind == "global":
            return GeneratedInstruction(
                    ast=Statement("sync; /* %s */" % comment),
                    implemented_domain=None)

        else:
            raise LoopyError("unknown barrier kind")

    def wrap_temporary_decl(self, decl, is_local):
        from cgen.ispc import ISPCUniform
        return ISPCUniform(decl)

    def get_global_arg_decl(self, name, shape, dtype, is_written):
        from loopy.codegen import POD  # uses the correct complex type
        from cgen import Const
        from cgen.ispc import ISPCUniformPointer, ISPCUniform

        arg_decl = ISPCUniformPointer(POD(self, dtype, name))

        if not is_written:
            arg_decl = Const(arg_decl)

        arg_decl = ISPCUniform(arg_decl)

        return arg_decl

    def get_value_arg_decl(self, name, shape, dtype, is_written):
        result = super(ISPCTarget, self).get_value_arg_decl(
                name, shape, dtype, is_written)

        from cgen import Reference, Const
        was_const = isinstance(result, Const)

        if was_const:
            result = result.subdecl

        if self.occa_mode:
            result = Reference(result)

        if was_const:
            result = Const(result)

        from cgen.ispc import ISPCUniform
        return ISPCUniform(result)

    def emit_sequential_loop(self, codegen_state, iname, iname_dtype,
            static_lbound, static_ubound, inner):
        ecm = codegen_state.expression_to_code_mapper

        from loopy.symbolic import aff_to_expr

        from loopy.codegen import wrap_in
        from pymbolic.mapper.stringifier import PREC_NONE
        from cgen import For

        return wrap_in(For,
                "uniform %s %s = %s"
                % (self.dtype_to_typename(iname_dtype),
                    iname, ecm(aff_to_expr(static_lbound), PREC_NONE, "i")),
                "%s <= %s" % (
                    iname, ecm(aff_to_expr(static_ubound), PREC_NONE, "i")),
                "++%s" % iname,
                inner)
    # }}}

# TODO: Generate launch code
# TODO: Vector types (element access: done)

# vim: foldmethod=marker
