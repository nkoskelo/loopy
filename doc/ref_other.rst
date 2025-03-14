Reference: Other Functionality
==============================

Auxiliary Data Types
--------------------

.. automodule:: loopy.typing

Obtaining Kernel Performance Statistics
---------------------------------------

.. automodule:: loopy.statistics

Controlling caching
-------------------

.. envvar:: LOOPY_NO_CACHE
.. envvar:: CG_NO_CACHE

    By default, loopy will cache (on disk) the result of various stages
    of code generation to speed up future code generation of the same kernel.
    By setting the environment variables :envvar:`LOOPY_NO_CACHE` or
    :envvar:`CG_NO_CACHE` to any
    string that :func:`pytools.strtobool` evaluates as ``True``, this caching
    is suppressed.


.. envvar:: LOOPY_ABORT_ON_CACHE_MISS

    If set to a string that :func:`pytools.strtobool` evaluates as ``True``,
    loopy will raise an exception if a cache miss occurs. This can be useful
    for debugging cache-related issues. For example, it can be used to automatically test whether caching is successful for a particular code, by setting this variable to ``True`` and re-running the code.


.. autofunction:: set_caching_enabled

.. autoclass:: CacheMode

Running Kernels
---------------

Use :class:`TranslationUnit.executor` to bind a translation unit
to execution resources, and then use :class:`ExecutorBase.__call__`
to invoke the kernel.

.. autoclass:: ExecutorBase

Automatic Testing
-----------------

.. autofunction:: auto_test_vs_ref

Troubleshooting
---------------

Printing :class:`LoopKernel` objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you're confused about things loopy is referring to in an error message or
about the current state of the :class:`LoopKernel` you are transforming, the
following always works::

    print(kernel)

(And it yields a human-readable--albeit terse--representation of *kernel*.)

.. autofunction:: get_dot_dependency_graph

.. autofunction:: show_dependency_graph

.. autofunction:: t_unit_to_python
