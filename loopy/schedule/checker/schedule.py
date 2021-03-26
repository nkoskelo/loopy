__copyright__ = "Copyright (C) 2019 James Stevens"

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

import islpy as isl
dt = isl.dim_type.set

__doc__ = """

.. data:: LIN_CHECK_IDENTIFIER_PREFIX

    The :class:`str` prefix for identifiers involved in linearization
    checking.

.. data:: LEX_VAR_PREFIX

    The :class:`str` prefix for the variables representing the
    dimensions in the lexicographic ordering used in a pairwise schedule. E.g.,
    a prefix of ``_lp_linchk_lex`` might yield lexicographic dimension
    variables ``_lp_linchk_lex0``, ``_lp_linchk_lex1``, ``_lp_linchk_lex2``.
    Cf.  :ref:`reserved-identifiers`.

.. data:: STATEMENT_VAR_NAME

    The :class:`str` name for the statement-identifying dimension of maps
    representing schedules and statement instance orderings.

.. data:: LTAG_VAR_NAME

    An array of :class:`str` names for map dimensions carrying values for local
    (intra work-group) thread identifiers in maps representing schedules and
    statement instance orderings.

.. data:: GTAG_VAR_NAME

    An array of :class:`str` names for map dimensions carrying values for group
    identifiers in maps representing schedules and statement instance orderings.

.. data:: BEFORE_MARK

    The :class:`str` identifier to be appended to input dimension names in
    maps representing schedules and statement instance orderings.

"""

LIN_CHECK_IDENTIFIER_PREFIX = "_lp_linchk_"
LEX_VAR_PREFIX = "%slex" % (LIN_CHECK_IDENTIFIER_PREFIX)
STATEMENT_VAR_NAME = "%sstmt" % (LIN_CHECK_IDENTIFIER_PREFIX)
LTAG_VAR_NAMES = []
GTAG_VAR_NAMES = []
for par_level in [0, 1, 2]:
    LTAG_VAR_NAMES.append("%slid%d" % (LIN_CHECK_IDENTIFIER_PREFIX, par_level))
    GTAG_VAR_NAMES.append("%sgid%d" % (LIN_CHECK_IDENTIFIER_PREFIX, par_level))
BEFORE_MARK = "'"


def _pad_tuple_with_zeros(tup, desired_length):
    return tup[:] + tuple([0]*(desired_length-len(tup)))


def _simplify_lex_dims(tup0, tup1):
    """Simplify a pair of lex tuples in order to reduce the complexity of
    resulting maps. Remove lex tuple dimensions with matching integer values
    since these do not provide information on relative ordering. Once a
    dimension is found where both tuples have non-matching integer values,
    remove any faster-updating lex dimensions since they are not necessary
    to specify a relative ordering.
    """

    new_tup0 = []
    new_tup1 = []

    # Loop over dims from slowest updating to fastest
    for d0, d1 in zip(tup0, tup1):
        if isinstance(d0, int) and isinstance(d1, int):

            # Both vals are ints for this dim
            if d0 == d1:
                # Do not keep this dim
                continue
            elif d0 > d1:
                # These ints inform us about the relative ordering of
                # two statements. While their values may be larger than 1 in
                # the lexicographic ordering describing a larger set of
                # statements, in a pairwise schedule, only ints 0 and 1 are
                # necessary to specify relative order. To keep the pairwise
                # schedules as simple and comprehensible as possible, use only
                # integers 0 and 1 to specify this relative ordering.
                # (doesn't take much extra time since we are already going
                # through these to remove unnecessary lex tuple dims)
                new_tup0.append(1)
                new_tup1.append(0)

                # No further dims needed to fully specify ordering
                break
            else:  # d1 > d0
                new_tup0.append(0)
                new_tup1.append(1)

                # No further dims needed to fully specify ordering
                break
        else:
            # Keep this dim without modifying
            new_tup0.append(d0)
            new_tup1.append(d1)

    if len(new_tup0) == 0:
        # Statements map to the exact same point(s) in the lex ordering,
        # which is okay, but to represent this, our lex tuple cannot be empty.
        return (0, ), (0, )
    else:
        return tuple(new_tup0), tuple(new_tup1)


class SpecialLexPointWRTLoop:
    """Strings specifying a particular position in a lexicographic
       ordering of statements relative to a loop.

    .. attribute:: PRE
       A :class:`str` indicating the last lexicographic point that
       precedes the loop.

    .. attribute:: FIRST
       A :class:`str` indicating the first lexicographic point in the
       first loop iteration (i.e., with the iname set to its min. val).

    .. attribute:: TOP
       A :class:`str` indicating the first lexicographic point in
       an arbitrary loop iteration.

    .. attribute:: BOTTOM
       A :class:`str` indicating the last lexicographic point in
       an arbitrary loop iteration.

    .. attribute:: LAST
       A :class:`str` indicating the last lexicographic point in the
       last loop iteration (i.e., with the iname set to its max val).

    .. attribute:: POST
       A :class:`str` indicating the first lexicographic point that
       follows the loop.
    """

    PRE = "pre"
    FIRST = "first"
    TOP = "top"
    BOTTOM = "bottom"
    LAST = "last"
    POST = "post"


def generate_pairwise_schedules(
        knl,
        lin_items,
        insn_id_pairs,
        loops_to_ignore=set(),
        return_schedules=False,
        ):
    r"""For each statement pair in a subset of all statement pairs found in a
    linearized kernel, determine the (relative) order in which the statement
    instances are executed. For each pair, describe this relative ordering with
    a pair of mappings from statement instances to points in a single
    lexicographic ordering (a ``pairwise schedule'').

    :arg knl: A preprocessed :class:`loopy.kernel.LoopKernel` containing the
        linearization items that will be used to create a schedule. This
        kernel will be used to get the domains associated with the inames
        used in the statements.

    :arg lin_items: A list of :class:`loopy.schedule.ScheduleItem`
        (to be renamed to `loopy.schedule.LinearizationItem`) containing
        all linearization items for which pairwise schedules will be
        created. To allow usage of this routine during linearization, a
        truncated (i.e. partial) linearization may be passed through this
        argument.

    :arg insn_id_pairs: A list containing pairs of instruction identifiers.

    :arg loops_to_ignore: A set of inames that will be ignored when
        determining the relative ordering of statements. This will typically
        contain concurrent inames tagged with the ``vec`` or ``ilp`` array
        access tags.

    :returns: A dictionary mapping each two-tuple of instruction identifiers
        provided in `insn_id_pairs` to a corresponding two-tuple containing two
        :class:`islpy.Map`\ s representing a pairwise schedule as two
        mappings from statement instances to lexicographic time, one for
        each of the two statements.
    """
    # TODO update docs now that we're returning SIOs

    from loopy.schedule import (EnterLoop, LeaveLoop, Barrier, RunInstruction)
    from loopy.kernel.data import (LocalIndexTag, GroupIndexTag)
    from loopy.schedule.checker.lexicographic_order_map import (
        create_lex_order_map,
        get_statement_ordering_map,
    )
    from loopy.schedule.checker.utils import (
        add_and_name_isl_dims,
        append_marker_to_strings,
        add_eq_isl_constraint_from_names,
    )
    slex = SpecialLexPointWRTLoop

    all_insn_ids = set().union(*insn_id_pairs)

    # First, use one pass through lin_items to generate a lexicographic
    # ordering describing the relative order of *all* statements represented by
    # all_insn_ids

    # For each statement, map the insn_id to a tuple representing points
    # in the lexicographic ordering containing items of :class:`int` or
    # :class:`str` :mod:`loopy` inames.
    stmt_inst_to_lex = {}

    # Keep track of the next tuple of points in our lexicographic
    # ordering, initially this as a 1-d point with value 0
    next_insn_lex_tuple = [0]

    for lin_item in lin_items:
        if isinstance(lin_item, EnterLoop):
            iname = lin_item.iname
            if iname in loops_to_ignore:
                continue

            # Increment next_insn_lex_tuple[-1] for statements in the section
            # of code after this EnterLoop.
            # (not technically necessary if no statement was added in the
            # previous section; gratuitous incrementing is counteracted
            # in the simplification step below)
            next_insn_lex_tuple[-1] += 1

            # Upon entering a loop, add one lex dimension for the loop variable,
            # add second lex dim to enumerate sections of code within new loop
            next_insn_lex_tuple.append(iname)
            next_insn_lex_tuple.append(0)

        elif isinstance(lin_item, LeaveLoop):
            if lin_item.iname in loops_to_ignore:
                continue

            # Upon leaving a loop,
            # pop lex dimension for enumerating code sections within this loop, and
            # pop lex dimension for the loop variable, and
            # increment lex dim val enumerating items in current section of code
            next_insn_lex_tuple.pop()
            next_insn_lex_tuple.pop()

            # Increment next_insn_lex_tuple[-1] for statements in the section
            # of code after this LeaveLoop.
            # (not technically necessary if no statement was added in the
            # previous section; gratuitous incrementing is counteracted
            # in the simplification step below)
            next_insn_lex_tuple[-1] += 1

        elif isinstance(lin_item, (RunInstruction, Barrier)):
            from loopy.schedule.checker.utils import (
                get_insn_id_from_linearization_item,
            )
            lp_insn_id = get_insn_id_from_linearization_item(lin_item)

            if lp_insn_id is None:
                assert isinstance(lin_item, Barrier)

                # Barriers without insn ids were inserted as a result of a
                # dependency. They don't themselves have dependencies. Ignore them.

                # FIXME: It's possible that we could record metadata about them
                # (e.g. what dependency produced them) and verify that they're
                # adequately protecting all statement instance pairs.

                continue

            # Only process listed insns, otherwise ignore
            if lp_insn_id in all_insn_ids:
                # Add item to stmt_inst_to_lex
                stmt_inst_to_lex[lp_insn_id] = tuple(next_insn_lex_tuple)

                # Increment lex dim val enumerating items in current section of code
                next_insn_lex_tuple[-1] += 1

        else:
            from loopy.schedule import (CallKernel, ReturnFromKernel)
            # No action needed for these types of linearization item
            assert isinstance(
                lin_item, (CallKernel, ReturnFromKernel))
            pass

        # To save time, stop when we've found all statements
        if len(stmt_inst_to_lex.keys()) == len(all_insn_ids):
            # TODO if combining blex map creation with this pass, cannot stop early
            break

    # Get dim names representing local/group axes for this kernel,
    # and get the dictionary that will be used later to create a
    # constraint requiring {par inames == par axes} in sched
    lid_lex_dim_names = set()
    gid_lex_dim_names = set()
    par_iname_constraint_dicts = []
    for iname in knl.all_inames():
        ltag = knl.iname_tags_of_type(iname, LocalIndexTag)
        if ltag:
            # assert len(ltag) == 1  # (should always be true)
            ltag_var = LTAG_VAR_NAMES[ltag.pop().axis]
            lid_lex_dim_names.add(ltag_var)
            # Represent constraint 'iname = ltag_var' in par_iname_constraint_dicts:
            par_iname_constraint_dicts.append({1: 0, iname: 1, ltag_var: -1})
            continue
        gtag = knl.iname_tags_of_type(iname, GroupIndexTag)
        if gtag:
            # assert len(gtag) == 1  # (should always be true)
            gtag_var = GTAG_VAR_NAMES[gtag.pop().axis]
            gid_lex_dim_names.add(gtag_var)
            # Represent constraint 'iname = gtag_var' in par_iname_constraint_dicts:
            par_iname_constraint_dicts.append({1: 0, iname: 1, gtag_var: -1})
            continue
    lid_lex_dim_names = sorted(lid_lex_dim_names)
    gid_lex_dim_names = sorted(gid_lex_dim_names)

    # {{{  Create blex ordering (may later be combined with pass above)

    # {{{ Determine which loops contain barriers

    loops_with_barriers = {"local": set(), "global": set()}
    current_inames = set()

    for lin_item in lin_items:
        if isinstance(lin_item, EnterLoop):
            current_inames.add(lin_item.iname)
        elif isinstance(lin_item, LeaveLoop):
            current_inames.remove(lin_item.iname)
        elif isinstance(lin_item, Barrier):
            loops_with_barriers[lin_item.synchronization_kind] |= current_inames
            # At this point we could technically skip ahead to next enterloop

    # }}}

    # {{{ Get upper and lower bound for each loop that contains a barrier
    # (Could try to combine this with pass below but would make things messy)

    iname_bounds_pwaff = {}
    for iname in loops_with_barriers["local"] | loops_with_barriers["global"]:
        # Get first and last vals for this iname
        bounds = knl.get_iname_bounds(iname)
        iname_bounds_pwaff[iname] = (
            bounds.lower_bound_pw_aff, bounds.upper_bound_pw_aff)

    # }}}

    conc_lex_dim_names = lid_lex_dim_names + gid_lex_dim_names

    def _collect_blex_ordering_info(sync_kind):

        # {{{ Construct blueprint for creating blex space and orderings

        stmt_inst_to_blex = {}  # map stmt instances to blex space
        iname_to_blex_dim = {}  # map from inames to corresponding blex space dim
        blex_exclusion_info = {}  # info for creating maps to exclude from blex order
        blex_map_params = set()  # params needed in blex map
        n_seq_blex_dims = 1  # num dims representing sequential order in blex space
        next_blex_pt = [0]  # next tuple of points in blex order

        for lin_item in lin_items:
            if isinstance(lin_item, EnterLoop):
                enter_iname = lin_item.iname
                if enter_iname in loops_with_barriers[sync_kind]:
                    # update next blex pt
                    pre_loop_blex_pt = next_blex_pt[:]
                    next_blex_pt[-1] += 1
                    next_blex_pt.append(enter_iname)
                    next_blex_pt.append(0)

                    # store tuples that will be used to create pairs
                    # that will later be subtracted from happens-before map
                    lbound = iname_bounds_pwaff[enter_iname][0]
                    first_iter_blex_pt = next_blex_pt[:]
                    first_iter_blex_pt[-2] = lbound
                    blex_exclusion_info[enter_iname] = {
                        slex.PRE: tuple(pre_loop_blex_pt),  # make sure to copy
                        slex.TOP: tuple(next_blex_pt),  # make sure to copy
                        slex.FIRST: tuple(first_iter_blex_pt),  # make sure to copy
                        }
                    blex_map_params |= set(lbound.get_var_names(dt.param))

            elif isinstance(lin_item, LeaveLoop):
                leave_iname = lin_item.iname
                if leave_iname in loops_with_barriers[sync_kind]:

                    # update max blex dims
                    n_seq_blex_dims = max(n_seq_blex_dims, len(next_blex_pt))
                    iname_to_blex_dim[leave_iname] = len(next_blex_pt)-2

                    # update next blex pt
                    pre_end_loop_blex_pt = next_blex_pt[:]
                    next_blex_pt.pop()
                    next_blex_pt.pop()
                    next_blex_pt[-1] += 1

                    # store tuples that will be used to create pairs
                    # that will later be subtracted from happens-before map
                    ubound = iname_bounds_pwaff[leave_iname][1]
                    last_iter_blex_pt = pre_end_loop_blex_pt[:]
                    last_iter_blex_pt[-2] = ubound
                    blex_exclusion_info[leave_iname][slex.BOTTOM] = tuple(
                        pre_end_loop_blex_pt)
                    blex_exclusion_info[leave_iname][slex.LAST] = tuple(
                        last_iter_blex_pt)
                    blex_exclusion_info[leave_iname][slex.POST] = tuple(
                        next_blex_pt)
                    # (make sure ^these are copies)
                    blex_map_params |= set(ubound.get_var_names(dt.param))

            elif isinstance(lin_item, RunInstruction):
                # Add item to stmt_inst_to_blex
                stmt_inst_to_blex[lin_item.insn_id] = tuple(next_blex_pt)
                # Don't increment blex dim val

            elif isinstance(lin_item, Barrier):
                # Increment blex dim val if the sync scope matches
                if lin_item.synchronization_kind == sync_kind:
                    next_blex_pt[-1] += 1

            else:
                from loopy.schedule import (CallKernel, ReturnFromKernel)
                # No action needed for these types of linearization item
                assert isinstance(
                    lin_item, (CallKernel, ReturnFromKernel))
                pass

        blex_map_params = sorted(blex_map_params)

        # At this point, some blex tuples may have more dimensions than others;
        # the missing dims are the fastest-updating dims, and their values should
        # be zero. Add them.
        for stmt, tup in stmt_inst_to_blex.items():
            stmt_inst_to_blex[stmt] = _pad_tuple_with_zeros(tup, n_seq_blex_dims)

        # }}}

        # Create names for the blex dimensions for sequential loops
        seq_blex_dim_names = [
            LEX_VAR_PREFIX+str(i) for i in range(n_seq_blex_dims)]
        seq_blex_dim_names_prime = append_marker_to_strings(
            seq_blex_dim_names, marker=BEFORE_MARK)

        blex_order_map = create_lex_order_map(
            dim_names=seq_blex_dim_names,
            in_dim_marker=BEFORE_MARK,
            )

        # Add lid/gid dims to lex order map
        blex_order_map = add_and_name_isl_dims(
            blex_order_map, dt.out, conc_lex_dim_names)
        blex_order_map = add_and_name_isl_dims(
            blex_order_map, dt.in_, append_marker_to_strings(conc_lex_dim_names))
        if sync_kind == "local":
            # Constrain gid vars to be equal
            for var_name in gid_lex_dim_names:
                blex_order_map = add_eq_isl_constraint_from_names(
                        blex_order_map, var_name, var_name+BEFORE_MARK)
        # (if sync_kind == "global", don't need constraints on lid/gid vars)

        iname_to_blex_var = {}
        for iname, dim in iname_to_blex_dim.items():
            iname_to_blex_var[iname] = seq_blex_dim_names[dim]
            iname_to_blex_var[iname+BEFORE_MARK] = seq_blex_dim_names_prime[dim]

        # Add params to blex map
        blex_order_map = add_and_name_isl_dims(
            blex_order_map, dt.param, blex_map_params)

        # Get a set representing blex_order_map space
        n_blex_dims = n_seq_blex_dims + len(conc_lex_dim_names)
        blex_set_template = isl.align_spaces(
            isl.Map("[ ] -> { [ ] -> [ ] }"), blex_order_map
            ).move_dims(
            dt.in_, n_blex_dims, dt.out, 0, n_blex_dims
            ).domain()
        blex_set_affs = isl.affs_from_space(blex_set_template.space)

        def _create_excluded_map_for_iname(iname, blueprint):
            # Note: blueprint[slex.FIRST] and blueprint[slex.LAST] contain pwaffs

            def _create_blex_set_from_tuple_pair(before, after, wrap_cond=False):

                # start with a set representing blex_order_map space
                blex_set = blex_set_template.copy()

                # add markers to inames in before tuple
                # (assume strings are the inames)
                before_prime = tuple(
                    v+BEFORE_MARK if isinstance(v, str) else v for v in before)
                before_padded = _pad_tuple_with_zeros(before_prime, n_seq_blex_dims)
                after_padded = _pad_tuple_with_zeros(after, n_seq_blex_dims)

                # assign vals to dims
                for dim_name, dim_val in zip(
                        seq_blex_dim_names_prime+seq_blex_dim_names,
                        before_padded+after_padded):
                    # (could exploit knowledge of content types of odd/even
                    # tuple dims to reduce conditionals but would be ugly
                    # and less robust)
                    if isinstance(dim_val, int):
                        # set idx to int val
                        blex_set &= blex_set_affs[dim_name].eq_set(
                            blex_set_affs[0]+dim_val)
                    elif isinstance(dim_val, str):
                        # assume this is an iname, set idx to corresponding blex var
                        blex_set &= blex_set_affs[dim_name].eq_set(
                            blex_set_affs[iname_to_blex_var[dim_val]])
                    else:
                        assert isinstance(dim_val, isl.PwAff)
                        pwaff_aligned = isl.align_spaces(dim_val, blex_set_affs[0])
                        # (doesn't matter which element of blex_set_affs we use^)
                        blex_set &= blex_set_affs[dim_name].eq_set(pwaff_aligned)

                if wrap_cond:
                    # i = i' + step
                    # TODO what about step sizes != 1?
                    blex_set &= blex_set_affs[iname_to_blex_var[iname]].eq_set(
                        blex_set_affs[iname_to_blex_var[iname+BEFORE_MARK]] + 1)

                return blex_set

            # enter loop case
            full_blex_set = _create_blex_set_from_tuple_pair(
                blueprint[slex.PRE], blueprint[slex.FIRST])
            # wrap loop case
            full_blex_set |= _create_blex_set_from_tuple_pair(
                blueprint[slex.BOTTOM], blueprint[slex.TOP], wrap_cond=True)
            # leave loop case
            full_blex_set |= _create_blex_set_from_tuple_pair(
                blueprint[slex.LAST], blueprint[slex.POST])

            # add cond to fix iteration value for surrounding loops (i = i')
            for surrounding_iname in blueprint[slex.PRE][1::2]:
                s_blex_var = iname_to_blex_var[surrounding_iname]
                full_blex_set &= blex_set_affs[s_blex_var].eq_set(
                    blex_set_affs[s_blex_var+BEFORE_MARK])

            # convert blex set back to map
            return isl.Map.from_domain(full_blex_set).move_dims(
                dt.out, 0, dt.in_, n_blex_dims, n_blex_dims)

        # subtract unwanted pairs from happens-before blex map
        maps_to_subtract = []
        for iname, subdict in blex_exclusion_info.items():
            maps_to_subtract.append(_create_excluded_map_for_iname(iname, subdict))

        if maps_to_subtract:
            # get union of maps
            map_to_subtract = maps_to_subtract[0]
            for other_map in maps_to_subtract[1:]:
                map_to_subtract |= other_map

            # get some closure
            map_to_subtract, closure_exact = map_to_subtract.transitive_closure()
            assert closure_exact  # TODO warn instead

            # subtract from blex order map
            blex_order_map = blex_order_map - map_to_subtract

        return (
            stmt_inst_to_blex,  # map stmt instances to blex space
            blex_order_map,
            seq_blex_dim_names,
            )

    (stmt_inst_to_lblex,
     lblex_order_map,
     seq_lblex_dim_names) = _collect_blex_ordering_info("local")
    (stmt_inst_to_gblex,
     gblex_order_map,
     seq_gblex_dim_names) = _collect_blex_ordering_info("global")

    # }}}  end blex order/map machinery

    # Second, create pairwise schedules for each individual pair of insns

    from loopy.schedule.checker.utils import (
        sorted_union_of_names_in_isl_sets,
        create_symbolic_map_from_tuples,
        insert_and_name_isl_dims,
    )

    def _get_map_for_stmt(
            insn_id, lex_points, int_sid, lex_dim_names):

        # Get inames domain for statement instance (a BasicSet)
        dom = knl.get_inames_domain(
            knl.id_to_insn[insn_id].within_inames)
        # (note that this domain may include inames that are
        # not in stmt.within_inames)

        # Create map space (an isl space in current implementation)
        # {('statement', <inames used in statement domain>) ->
        #  (lexicographic ordering dims)}
        dom_inames_ordered = sorted_union_of_names_in_isl_sets([dom])

        in_names_sched = [STATEMENT_VAR_NAME] + dom_inames_ordered[:]
        sched_space = isl.Space.create_from_names(
            isl.DEFAULT_CONTEXT,
            in_=in_names_sched,
            out=lex_dim_names,
            params=[],
            )

        # Insert 'statement' dim into domain so that its space allows
        # for intersection with sched map later
        dom_to_intersect = insert_and_name_isl_dims(
                dom, dt.set, [STATEMENT_VAR_NAME], 0)

        # Each map will map statement instances -> lex time.
        # Right now, statement instance tuples consist of single int.
        # Add all inames from domains to each map domain tuple.
        tuple_pair = [(
            (int_sid, ) + tuple(dom_inames_ordered),
            lex_points
            )]

        # Note that lex_points may have fewer dims than the out-dim of sched_space
        # if sched_space includes concurrent lid/gid dims. This is okay because
        # the following symbolic map creation step, when assigning dim values,
        # zips the space dims with the lex tuple, and any leftover lid/gid dims
        # will not be assigned a value yet, which is what we want.

        # Create map
        sched_map = create_symbolic_map_from_tuples(
            tuple_pairs_with_domains=zip(tuple_pair, [dom_to_intersect, ]),
            space=sched_space,
            )

        # Set inames equal to relevant gid/lid var names
        for constraint_dict in par_iname_constraint_dicts:
            sched_map = sched_map.add_constraint(
                isl.Constraint.eq_from_names(sched_map.space, constraint_dict))

        return sched_map

    pairwise_schedules = {}
    for insn_ids in insn_id_pairs:
        # Determine integer IDs that will represent each statement in mapping
        # (dependency map creation assumes sid_before=0 and sid_after=1, unless
        # before and after refer to same stmt, in which case sid_before=sid_after=0)
        int_sids = [0, 0] if insn_ids[0] == insn_ids[1] else [0, 1]

        # {{{  Create SIO for intra-thread case (lid0' == lid0, etc)

        # Simplify tuples to the extent possible ------------------------------------

        lex_tuples = [stmt_inst_to_lex[insn_id] for insn_id in insn_ids]

        # At this point, one of the lex tuples may have more dimensions than another;
        # the missing dims are the fastest-updating dims, and their values should
        # be zero. Add them.
        max_lex_dims = max([len(lex_tuple) for lex_tuple in lex_tuples])
        lex_tuples_padded = [
            _pad_tuple_with_zeros(lex_tuple, max_lex_dims)
            for lex_tuple in lex_tuples]

        # Now generate maps from the blueprint --------------------------------------

        lex_tuples_simplified = _simplify_lex_dims(*lex_tuples_padded)

        # Create names for the output dimensions for sequential loops
        seq_lex_dim_names = [
            LEX_VAR_PREFIX+str(i) for i in range(len(lex_tuples_simplified[0]))]

        intra_thread_sched_maps = [
            _get_map_for_stmt(
                insn_id, lex_tuple, int_sid,
                seq_lex_dim_names+conc_lex_dim_names)
            for insn_id, lex_tuple, int_sid
            in zip(insn_ids, lex_tuples_simplified, int_sids)
            ]

        # Create lex order maps and SIOs here (rather than returning schedules
        # and lex maps separately and combining them outside function to get
        # SIOs) to avoid passing extra info around. Don't want to, e.g.,
        # examine the schedule tuple in separate func to re-determine which
        # parallel dims are used. (could simplify everything by always using
        # all dims..., which would make maps more complex than necessary)
        lex_order_map = create_lex_order_map(
            dim_names=seq_lex_dim_names,
            in_dim_marker=BEFORE_MARK,
            )

        # Add lid/gid dims to lex order map
        lex_order_map = add_and_name_isl_dims(
            lex_order_map, dt.out, conc_lex_dim_names)
        lex_order_map = add_and_name_isl_dims(
            lex_order_map, dt.in_, append_marker_to_strings(conc_lex_dim_names))
        # Constrain lid/gid vars to be equal
        for var_name in conc_lex_dim_names:
            lex_order_map = add_eq_isl_constraint_from_names(
                lex_order_map, var_name, var_name+BEFORE_MARK)

        # Create statement instance ordering,
        # maps each statement instance to all statement instances occuring later
        sio_seq = get_statement_ordering_map(
            *intra_thread_sched_maps,  # note, func accepts exactly two maps
            lex_order_map,
            before_marker=BEFORE_MARK,
            )

        # }}}

        # {{{  Create SIOs for intra-group case (gid0' == gid0, etc)

        # TODO finish separating lid stuff from gid stuff

        # Use *unsimplified* lex tuples with blex map, which have already been padded

        lblex_tuples_padded = [stmt_inst_to_lblex[insn_id] for insn_id in insn_ids]

        lconc_sched_maps = [
            _get_map_for_stmt(
                insn_id, lblex_tuple, int_sid,
                seq_lblex_dim_names+conc_lex_dim_names)  # conc names same for all
            for insn_id, lblex_tuple, int_sid
            in zip(insn_ids, lblex_tuples_padded, int_sids)
            ]

        # Create statement instance ordering
        sio_lconc = get_statement_ordering_map(
            *lconc_sched_maps,  # note, func accepts exactly two maps
            lblex_order_map,
            before_marker=BEFORE_MARK,
            )

        # TODO use func to avoid duplicated code here:

        gblex_tuples_padded = [stmt_inst_to_gblex[insn_id] for insn_id in insn_ids]

        gconc_sched_maps = [
            _get_map_for_stmt(
                insn_id, gblex_tuple, int_sid,
                seq_gblex_dim_names+conc_lex_dim_names)  # conc names same for all
            for insn_id, gblex_tuple, int_sid
            in zip(insn_ids, gblex_tuples_padded, int_sids)
            ]

        # Create statement instance ordering
        sio_gconc = get_statement_ordering_map(
            *gconc_sched_maps,  # note, func accepts exactly two maps
            gblex_order_map,
            before_marker=BEFORE_MARK,
            )

        # }}}

        if return_schedules:
            # Store sched maps along with SIOs
            # (currently helpful for testing; also could be desired by a user)
            pairwise_schedules[tuple(insn_ids)] = (
                (sio_seq, tuple(intra_thread_sched_maps), ),
                (sio_lconc, tuple(lconc_sched_maps), ),
                (sio_gconc, tuple(gconc_sched_maps), ),
                )
        else:
            # Store SIOs
            pairwise_schedules[tuple(insn_ids)] = (sio_seq, sio_lconc, sio_gconc)

    return pairwise_schedules