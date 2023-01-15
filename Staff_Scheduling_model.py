import sys
import random
from ortools.linear_solver import pywraplp


def val_obj(x):
    return x.Objective().Value()


def val_sol(x):
    if type(x) is not list:
        if x is None:
            return 0
        return x.SolutionValue()
    elif type(x) is list:
        ret_list = []
        for e in x:
            ret_list.append(val_sol(e))
        return ret_list


def k_out_of_n(solver, k, x_info, rel='=='):
    len_n = len(x_info)
    binary = sum(x_info[i].Lb() == 0 for i in range(len_n)) == len_n and sum(x_info[i].Ub() == 1 for i in range(len_n)) == len_n
    if binary:
        l_new = x_info
    else:
        l_new = [solver.IntVar(0, 1, '') for i in range(len_n)]
        for i in range(len_n):
            if x_info[i].Ub() > 0:
                solver.Add(x_info[i] <= x_info[i].Ub() * l_new[i])
            else:
                solver.Add(x_info[i] >= x_info[i].Lb() * l_new[i])
    S = sum(l_new[i] for i in range(len_n))
    if rel == '==' or rel == '=':
        solver.Add(S == k)
    elif rel == '>=':
        solver.Add(S >= k)
    else:
        solver.Add(S <= k)
    return l_new


def box_bounds(a, x, b):
    Bounds, n = [None, None], len(a)
    s = pywraplp.Solver('Box', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
    xx = [s.NumVar(x[i].Lb(), x[i].Ub(), '') for i in range(n)]
    S = s.Sum([-b] + [a[i] * xx[i] for i in range(n)])
    s.Maximize(S)
    rc = s.Solve()
    Bounds[1] = None if rc != 0 else val_obj(s)
    s.Minimize(S)
    s.Solve()
    Bounds[0] = None if rc != 0 else val_obj(s)
    return Bounds


def solver_new(name, integer=False):
    return pywraplp.Solver(name,
                           pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING if integer else pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)


def raise_reify(s, d, x, c, new_delta=None, rel='<=', bounds=None, ep=1):
    len_n = len(d)
    if new_delta is None:
        new_delta = s.IntVar(0, 1, '')
    if bounds is None:
        bounds = box_bounds(d, x, c)
    if rel == '<=':
        s.Add(sum(d[i] * x[i] for i in range(len_n)) >= c + bounds[0] * new_delta + ep * (1 - new_delta))
    if rel == '>=':
        s.Add(sum(d[i] * x[i] for i in range(len_n)) <= c + bounds[1] * new_delta - ep * (1 - new_delta))
    elif rel == '==':
        gm = [s.IntVar(0, 1, '') for _ in range(2)]
        s.Add(sum(d[i] * x[i] for i in range(len_n)) >= c + bounds[0] * gm[0] + ep * (1 - gm[0]))
        s.Add(sum(d[i] * x[i] for i in range(len_n)) <= c + bounds[1] * gm[1] - ep * (1 - gm[1]))
        s.Add(gm[0] + gm[1] - 1 == new_delta)
    return new_delta


def reify(s, a, x, b, d=None, rel='<=', bnds=None, ep=1):
    return raise_reify(s, a, x, b, reify_force(s, a, x, b, d, rel, bnds), rel, bnds, ep)


def reify_force(s, d, x, c, new_delta=None, rel='<=', bounds=None):
    len_n = len(d)
    if new_delta is None:
        new_delta = s.IntVar(0, 1, '')
    if bounds is None:
        bounds = box_bounds(d, x, c)
    if rel in ['<=', '==']:
        sum_temp = 0
        for i in range(len_n):
            sum_temp += d[i] * x[i]
        s.Add(sum_temp <= c + bounds[1] * (1 - new_delta))
    elif rel in ['>=', '==']:
        sum_temp = 0
        for i in range(len_n):
            sum_temp += d[i] * x[i]
        s.Add(sum_temp >= c + bounds[0] * (1 - new_delta))
    return new_delta


from random import randint


# generates sections for each course id
def generate_sections_table(n):
    sec_lists = []
    section = 0
    for i in range(n):
        for j in range(randint(1, 4)):
            random_int = randint(1, 20)
            sec_lists_inner = [section, i, random_int]
            sec_lists.append(sec_lists_inner)
            section = section + 1
    return sec_lists, section


# instructor_count,sets_count,course_count,pairs_count
# generating table7-9
def generate_instructor_preference_table(m, n, p, pp):
    R = []
    for i in range(m):
        col_3 = []
        col_4 = []
        col_5 = []
        for _ in range(p):
            rand_3 = randint(0, 1) * randint(-10, 10)
            col_3.append(rand_3)
        for _ in range(n):
            rand_4 = randint(0, 1) * randint(-10, 10)
            col_4.append(rand_4)
        for _ in range(pp):
            rand_5 = randint(0, 1) * randint(-10, 10)
            col_5.append(rand_5)
        RR = [i, [randint(1, 2), randint(2, 3)], col_3, col_4, col_5]

        R.append(RR)
    return R


# generates preference sets for 6
# sections = 15, # of preference sets = 6
# generating table 7-10
def generate_preference_sets_table(n, ns):
    pref_set = []
    for i in range(ns):
        inner_list = []
        for j in range(n):
            # either add the section or not based on the binary choice.
            if randint(0, 1):
                inner_list.append((j))
        pref_set_inner = [i, inner_list]
        pref_set.append(pref_set_inner)
    return pref_set


# pairs_count,section_count
# pairs = 2 (number of rows - pairs count), section = 15
def generate_preference_pairs_table(pp, n):
    pair_lists = []
    for i in range(pp):
        q_num = 4
        c_0 = 0
        pair_inner_list = []
        for j in range(q_num):
            c_0 = randint(c_0, int(3 * n / q_num))
            c1 = randint(c_0 + 1, n - 1)
            if (c_0, c1) not in pair_inner_list:
                pair_inner_list.append((c_0, c1))
        pair_inner_list.sort()
        pair_lists.append([i, pair_inner_list])
    return pair_lists


from ortools.linear_solver import pywraplp


# tables for sections, instructors, preference sets, pairs
def model_solve(S, I, R, P):
    s = solver_new('Staff Scheduling', True)
    nbS, nbI, sets_count, pairs_count, nbC = len(S), len(I), len(R), len(P), S[-1][1] + 1
    # print(nbS, nbI, sets_count, pairs_count, nbC)
    nbT = 1 + max(e[2] for e in S)
    x = []
    for _ in range(nbI):
        x_inner = []
        for _ in range(nbS):
            x_inner.append(s.IntVar(0, 1, ''))
        x.append(x_inner)

    z = []
    for _ in range(nbI):
        inst_list = []
        for p in range(pairs_count):
            pair_list = []
            for _ in range(len(P[p][1])):
                pair_list.append(s.IntVar(0, 1, ''))
            inst_list.append(pair_list)
        z.append(inst_list)

    for j in range(nbS):
        k_out_of_n(s, 1, [x[i][j] for i in range(nbI)], '<=')

    for i in range(nbI):
        sum_temp = 0
        for j in range(nbS):
            sum_temp += x[i][j]
        s.Add(sum_temp >= I[i][1][0])
        s.Add(sum_temp <= I[i][1][1])
        for t in range(nbT):
            inner_list = []
            for j in range(nbS):
                if S[j][2] == t:
                    inner_list.append(x[i][j])

            k_out_of_n(s, 1, inner_list, '<=')
    # considering course preference
    WC = 0
    for i in range(nbI):
        for j in range(nbS):
            for c in range(nbC):
                if S[j][1] == c:
                    WC += x[i][j] * I[i][2][c]

    # considering time preference
    WR = 0
    for r in range(sets_count):
        for i in range(nbI):
            sum_j = 0
            for j in R[r][1]:
                sum_j += x[i][j]
            WR += I[i][3][r] * sum_j

    for i in range(nbI):
        for p in range(pairs_count):
            if I[i][4][p] != 0:
                for k in range(len(P[p][1])):
                    reify(s, [1, 1], [x[i][P[p][1][k][0]], x[i][P[p][1][k][1]]], 2, z[i][p][k], '>=')

    WP = 0
    for i in range(nbI):
        for p in range(pairs_count):
            for k in range(len(P[p][1])):
                if I[i][4][p] != 0:
                    WP += z[i][p][k] * I[i][4][p]


    s.Maximize(WC + WR + WP)
    rc, xs, xs_old = s.Solve(), [], []
    for i in range(nbI):
        xs_inner = []
        for j in range(nbS):
            if val_sol(x[i][j]) > 0:
                sum1 = 0
                for r in range(sets_count):
                    if j in R[r][1]:
                        sum1 += I[i][3][r]
                sum2 = 0
                for p in range(pairs_count):
                    for k in range(len(P[p][1])):
                        if j in P[p][1][k]:
                            sum2 += val_sol(z[i][p][k]) * I[i][4][p] / 2
                xs_inner.append([j, (I[i][2][S[j][1]], sum1, sum2)])

        xs.append([i, xs_inner])

    return rc, val_sol(x), xs, val_obj(s)


import copy

def matrix_wrap(matrix_list, left, head):
    matrix_list_copy = copy.deepcopy(matrix_list)
    # m, n = len(matrix_list_copy), len(matrix_list_copy[0])
    for i in range(len(left)):
        matrix_list_copy[i].insert(0, left[i])
    if head != None:
        if len(head) < len(matrix_list_copy[0]):
            matrix_list_copy.insert(0, [''] + head)
        else:
            matrix_list_copy.insert(0, head)
    return matrix_list_copy


def matrix_format(matrix_list, zeroes=False, decimals=4):
    matrix_list_copy = copy.deepcopy(matrix_list)
    for i in range(len(matrix_list)):
        for j in range(len(matrix_list[i])):
            element = matrix_list_copy[i][j]
            if type(element) == int:
                if element or zeroes:
                    element = '{0:4d}'.format(element)
                else:
                    element = ''
            elif type(element) == float:
                if element or zeroes:
                    if decimals == 4:
                        element = '{0:.4f}'.format(element)
                    elif decimals == 3:
                        element = '{0:.3f}'.format(element)
                    elif decimals == 2:
                        element = '${0:.2f}'.format(element)
                    elif decimals == 1:
                        element = '{0:.1f}'.format(element)
                    elif decimals == 0:
                        element = '{0:.0f}'.format(element)
                else:
                    element = ''
            matrix_list_copy[i][j] = element
    return matrix_list_copy


def matrix_print(M, zeroes=False, decimals=4):
    rows = matrix_format(M, zeroes, decimals)
    for r in rows:
        l_str = ''
        for i in range(len(r)):
            l_str = l_str + str(r[i])
            if i < len(r) - 1:
                l_str = l_str + ','
        print(l_str)

def main():
    sets_count = 6
    instructor_count = 5
    pairs_count = 2
    course_count = 7
    random.seed()
    S, section_count = generate_sections_table(course_count)
    R = generate_preference_sets_table(section_count, sets_count)
    I = generate_instructor_preference_table(instructor_count, sets_count, course_count, pairs_count)
    P = generate_preference_pairs_table(pairs_count, section_count)
    new_rc, val_sol_x, new_xs, obj_val = model_solve(S, I, R, P)
    XS_primary_list = []
    for i in range(len(new_xs)):
        XS_primary_list.append([new_xs[i][0],
                   ['{0:2}'.format(ele[0]) +
                    ' : (' + '{0:2}'.format(ele[1][0]) +
                    ' ' + '{0:2}'.format(
                       ele[1][1]) + ' ' +
                    '{0:2}'.format(ele[1][2]) + ')' for ele in new_xs[i][1]]])
    matrix_print(matrix_wrap(XS_primary_list, [], ['Instructor', 'Section (WC WR WP)']), True, 1)


main()
