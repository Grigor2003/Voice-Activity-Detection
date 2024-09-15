def check_req(value, required):
    if required and value is None:
        raise LookupError("At least one of required arguments was not given")
    else:
        return value


def at_least_one_of(args):
    if all(v is None for v in args):
        raise LookupError("One of required arguments was not given")


def typecheck(v, tp):
    if isinstance(tp, (list, tuple)):
        return type(v) in tp
    return type(v) is tp


def is_type_of(value, tp=str, req=True):
    check_req(value, req)
    if value is not None and not typecheck(value, tp):
        raise TypeError(f"{value} is not a valid {tp}")
    else:
        return value


def with_range(value, fr=0, to=1, tp=(int, float), req=True):
    if tp is not None:
        is_type_of(value, tp, req)
    else:
        check_req(value, req)
    if value is None:
        return value
    if fr <= value <= to:
        return value
    raise ValueError(f"{value} is out of range ({fr} to {to})")


def parse_range(lst, first_range, second_range, order=True, req=True):
    check_req(lst, req)
    if lst is None:
        return None, None
    try:
        if len(lst) != 2:
            raise IndexError("Range must have exactly 2 numeric items")
        s, e = lst[0], lst[1]
    except Exception as err:
        raise TypeError(
            f"Range must be in the form [start, end]. The form you specified caused the following error: {str(err)}")
    with_range(s, *first_range)
    with_range(e, *second_range)
    if order and s > e:
        raise TypeError(
            f"In [start, end], start({s}) cannot be greater than end({e})")
    return s, e


def parse_linspace(lst, first_range, second_range, count_range, order=True, req=True):
    check_req(lst, req)
    if lst is None:
        return None, None
    try:
        if len(lst) != 3:
            raise IndexError("Range must have exactly 3 numeric items")
        s, e, c = lst[0], lst[1], lst[2]
    except Exception as err:
        raise TypeError(
            f"Range must be in the form [start, end, count]. The form you specified caused the following error: {str(err)}")
    with_range(s, *first_range)
    with_range(e, *second_range)
    with_range(c, *count_range)
    if order and s > e:
        raise TypeError(
            f"In [start, end, count], start({s}) cannot be greater than end({e})")
    return s, e, c
