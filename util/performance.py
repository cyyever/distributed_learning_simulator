from itertools import chain
from sys import getsizeof


def total_size(o, handlers={}):
    """Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """

    def dict_handler(d):
        return chain.from_iterable(d.items())

    all_handlers = {
        tuple: iter,
        list: iter,
        dict: dict_handler,
        set: iter,
        frozenset: iter,
    }
    all_handlers.update(handlers)  # user handlers take precedence
    seen = set()  # track which object id's have already been seen
    default_size = getsizeof(0)  # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:  # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)
        if isinstance(o, float | int | bool | str):
            return s
        if isinstance(o, torch.Tensor):
            return o.element_size() * o.nelement()
        for attr in dir(o):
            if attr.startswith("__"):
                continue
            if hasattr(o, attr):
                value = getattr(o, attr)
                if hasattr(value, "__call__"):
                    continue
                # print("attr is", attr, type(value))
                s += sizeof(value)

        return s

    return sizeof(o)
