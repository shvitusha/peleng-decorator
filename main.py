from functools import wraps


def cached_result(func):
    cache = {"count": 0, "result": None}

    @wraps(func)
    def wrapper(*args, **kwargs):
        if cache["count"] < 3:
            if cache["result"] is not None:
                result = cache["result"]
                #print("\nsave result")
            else:
                result = func(*args, **kwargs)
                cache["result"] = result
                #print("\ncalc")
            cache["count"] += 1
        else:
            cache["count"] = 1
            result = func(*args, **kwargs)
            cache["result"] = result
            #print("\nrepeat save result")
        return result

    return wrapper


@cached_result
def my_function(x):
    return x ** x + 20


print(my_function(6))
print(my_function(6))
print(my_function(6))
print(my_function(6))
