from utils.tube import Tube

def setupCTCR(
        n: int = 3,
        ls: list[float] = [447*1e-3, 277*1e-3, 94*1e-3],
        lc: list[float] = [58*1e-3, 94*1e-3, 91*1e-3],
        ro: list[float] = [1.00/2*1e-3, 1.75/2*1e-3, 2.37/2*1e-3],
        ri: list[float] = [0.80/2*1e-3, 1.25/2*1e-3, 2.00/2*1e-3],
        k: list[float] = [1e3/55.6, 1e3/60, 1e3/160],
        nu: float = 0.3,
        E: float = 60e09
    ) -> list[Tube]:

    return [Tube(ls[i], lc[i], ro[i], ri[i], k[i], nu, E) for i in range(n)]

