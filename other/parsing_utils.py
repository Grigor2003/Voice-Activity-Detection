def check_req(value, required):
    if required and value is None:
        raise LookupError("At least one of required arguments was not given")
    else:
        return value


def at_least_one_of(args):
    if all(v is None for v in args):
        raise LookupError("One of required arguments was not given")


def is_type_of(value, tp=str, req=True):
    check_req(value, req)
    if value is not None and not isinstance(value, tp):
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


def parse_list(lst, first, second, order=True, req=True):
    check_req(lst, req)
    if lst is None:
        return None, None
    try:
        s, e = map(int, lst)
        with_range(s, *first)
        with_range(e, *second)
        if order and s > e:
            raise TypeError(
                f"{lst} cannot be greater than {e}")
        return s, e
    except Exception as err:
        raise TypeError(
            f"Range must be in the form [start, end]. The form you specified caused the following error: {str(err)}")
