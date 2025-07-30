# define problems for PDE

from FKproblem import FKproblem
from GBMproblem import GBMproblem
from PoissonProblem import PoissonProblem
from SimpleODEProblem import SimpleODEProblem
from VarPoiProblem import VarPoiProblem
from HeatProblem import HeatProblem
from BurgerProblem import BurgerProblem
from varFKproblem import varFKproblem
from DarcyProblem import DarcyProblem

from PoissonHyper import PoissonHyper

# neural operator
from PointProcNeuralOp import PointProcessOperatorLearning
from FKOpLproblem import FKOperatorLearning
from VarPoiDeepONet import VarPoiDeepONet

# Bayesian
from PoissonBayesian import PoissonBayesian
from PointProcess import PointProcess
from PoissonFunBayesian import PoissonFunBayesian
from VarPoiBayesProblem import VarPoiBayesProblem
from NonlinearPoisson import NonlinearPoisson
from Darcy1dBayes import Darcy1dBayes
from Darcy2dBayesProblem import Darcy2dBayes

def create_pde_problem(pde_opts):
    problem_type = pde_opts['problem']
    
    problem_classes = {
        'poisson': PoissonProblem,
        'poissonbayesian': PoissonBayesian,
        'funbayesian': PoissonFunBayesian,
        'poissonhyper': PoissonHyper,
        'simpleode': SimpleODEProblem,
        'fk': FKproblem,
        'gbm': GBMproblem,
        'poivar': VarPoiProblem,
        'varfk': varFKproblem,
        'heat': HeatProblem,
        'burger': BurgerProblem,
        'darcy': DarcyProblem,
        'pointprocess': PointProcess,
        'pointprocessop': PointProcessOperatorLearning,
        'fkop': FKOperatorLearning,
        'varpoiop': VarPoiDeepONet,
        'varpoibayes': VarPoiBayesProblem,
        'nonlinearpoisson': NonlinearPoisson,
        'darcy1dbayes': Darcy1dBayes,
        'darcy2dbayes': Darcy2dBayes,
    }
    
    if problem_type in problem_classes:
        return problem_classes[problem_type](**pde_opts)
    else:
        raise ValueError(f'Unknown problem type: {problem_type}')

# if __name__ == "__main__":
    # simple visualization of the data set
    