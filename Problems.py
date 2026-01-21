import importlib

def create_pde_problem(pde_opts):
    problem_type = pde_opts['problem']
    
    # Maps problem_type string to (module_name, class_name)
    problem_map = {
        'poisson': ('PoissonProblem', 'PoissonProblem'),
        'seppoisson': ('PoissonSepProblem', 'PoissonSepProblem'),
        'poissonbayesian': ('PoissonBayesian', 'PoissonBayesian'),
        'funbayesian': ('PoissonFunBayesian', 'PoissonFunBayesian'),
        'poissonhyper': ('PoissonHyper', 'PoissonHyper'),
        'simpleode': ('SimpleODEProblem', 'SimpleODEProblem'),
        'fk': ('FKproblem', 'FKproblem'),
        'gbm': ('GBMproblem', 'GBMproblem'),
        'gbmbayes': ('GBMBayesProblem', 'GBMBayesProblem'),
        'poivar': ('VarPoiProblem', 'VarPoiProblem'),
        'pvneumann': ('VarPoiNeumannProblem', 'VarPoiNeumannProblem'),
        'varfk': ('varFKproblem', 'varFKproblem'),
        'heat': ('HeatProblem', 'HeatProblem'),
        'burger': ('BurgerProblem', 'BurgerProblem'),
        'darcy': ('DarcyProblem', 'DarcyProblem'),
        'pointprocess': ('PointProcess', 'PointProcess'),
        'fkop': ('FKOpLproblem', 'FKOperatorLearning'),
        'varpoiop': ('VarPoiDeepONet', 'VarPoiDeepONet'),
        'varpoibayes': ('VarPoiBayesProblem', 'VarPoiBayesProblem'),
        'nonlinearpoisson': ('NonlinearPoisson', 'NonlinearPoisson'),
        'darcy1dbayes': ('Darcy1dBayes', 'Darcy1dBayes'),
        'darcy2dbayes': ('Darcy2dBayesProblem', 'Darcy2dBayes'),
    }
    
    if problem_type in problem_map:
        module_name, class_name = problem_map[problem_type]
        try:
            module = importlib.import_module(module_name)
            ProblemClass = getattr(module, class_name)
            return ProblemClass(**pde_opts)
        except ImportError:
            raise ValueError(f"Could not import module '{module_name}' for problem type '{problem_type}'.")
        except AttributeError:
            raise ValueError(f"Could not find class '{class_name}' in module '{module_name}'.")
    else:
        raise ValueError(f'Unknown problem type: {problem_type}')