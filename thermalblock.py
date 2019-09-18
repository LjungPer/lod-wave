from pymor.basic import thermal_block_problem, discretize_stationary_cg, CoerciveRBReductor, ExpressionParameterFunctional, greedy

p = thermal_block_problem(num_blocks=(3, 2))
d, d_data = discretize_stationary_cg(p, diameter=1./100)

U = d.solve([1.0, 0.1, 0.3, 0.1, 0.2, 1.0])

reductor = CoerciveRBReductor(d, product=d.h1_0_semi_product, \
    coercivity_estimator=ExpressionParameterFunctional('min(diffusion)', d.parameter_type))

samples = d.parameter_space.sample_uniformly(4)
print(samples[0])

greedy_data = greedy(d, reductor, samples, use_estimator=True, max_extensions=32)