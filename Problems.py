# define problems for PDE

from FKproblem import FKproblem
from GBMproblem import GBMproblem
from PoissonProblem import PoissonProblem
from SimpleODEProblem import SimpleODEProblem
from LorenzProblem import LorenzProblem
from VarPoiProblem import VarPoiProblem
from HeatProblem import HeatProblem
from BurgerProblem import BurgerProblem
from varFKproblem import varFKproblem
from DarcyProblem import DarcyProblem

from PoissonProblem2 import PoissonProblem2
from PoissonHyper import PoissonHyper
from PointProcess import PointProcess
from FKOpLproblem import FKOperatorLearning
from VarPoiDeepONet import VarPoiDeepONet

def create_pde_problem(pde_opts):
    problem_type = pde_opts['problem']
    
    problem_classes = {
        'poisson': PoissonProblem,
        'poisson2': PoissonProblem2,
        'poissonhyper': PoissonHyper,
        'lorenz': LorenzProblem,
        'simpleode': SimpleODEProblem,
        'fk': FKproblem,
        'gbm': GBMproblem,
        'poivar': VarPoiProblem,
        'varfk': varFKproblem,
        'heat': HeatProblem,
        'burger': BurgerProblem,
        'darcy': DarcyProblem,
        'fkop': FKOperatorLearning,
        'varpoiop': VarPoiDeepONet
    }
    
    if problem_type in problem_classes:
        return problem_classes[problem_type](**pde_opts)
    else:
        raise ValueError(f'Unknown problem type: {problem_type}')

# if __name__ == "__main__":
    # simple visualization of the data set
    