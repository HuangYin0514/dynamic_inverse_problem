from .el import Euler, ImplicitEuler
from .rk import RK4, RK4_high_order

__factory = {
    'RK4': RK4,
    'RK4_high_order': RK4_high_order,
    'Euler': Euler,
    'ImplicitEuler': ImplicitEuler,
}


def ODEIntegrate(func, t0, t1, dt, y0, method, *args, **kwargs):
    if method not in __factory.keys():
        raise ValueError('solver \'{}\' is not implemented'.format(method))
    results = __factory[method](func, t0, t1, dt, y0, *args, **kwargs).get_results(*args, **kwargs)
    return results
